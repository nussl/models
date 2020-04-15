import nussl
import torch
import numpy as np
import gin
import ignite

@gin.configurable
def add_train_handlers(engine, model, scheduler, optimizer, 
                       train_closure, device, handler_names):
    for handler_name in handler_names:
        if handler_name == "add_clip_gradient_handler":
            add_clip_gradient_handler(engine, model)
        elif handler_name == "add_lr_scheduler_handler":
            add_lr_scheduler_handler(engine, scheduler)
        elif handler_name == "add_autoclip_gradient_handler":
            add_autoclip_gradient_handler(engine, model)
        elif handler_name == "add_inspect_gradient":
            add_inspect_gradient(engine, model)
        elif handler_name == "add_auto_balance_loss":
            add_auto_balance_loss(engine, train_closure, device)

def _get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm 

@gin.configurable
def add_clip_gradient_handler(engine, model, clip_value):
    @engine.on(nussl.ml.train.BackwardsEvents.BACKWARDS_COMPLETED)
    def clip_gradient(engine):
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

def add_lr_scheduler_handler(engine, scheduler):
    @engine.on(nussl.ml.train.ValidationEvents.VALIDATION_COMPLETED)
    def step_scheduler(engine):
        val_loss = engine.state.epoch_history['validation/loss'][-1]
        scheduler.step(val_loss)

@gin.configurable
def add_inspect_gradient(engine, model):
    @engine.on(nussl.ml.train.BackwardsEvents.BACKWARDS_COMPLETED)
    def inspect_gradient(engine):
        obs_grad_norm = _get_grad_norm(model)
        if 'grad_norm' not in engine.state.iter_history:
            engine.state.iter_history['grad_norm'] = []
        engine.state.iter_history['grad_norm'].append(obs_grad_norm)

@gin.configurable
def add_auto_balance_loss(engine, train_closure, device, 
                          ref_percentile=100, update_frequency=1):
    n_losses = len(train_closure.losses)
    scale = 1 / n_losses
    loss_weights = torch.nn.ParameterList([
        torch.nn.Parameter(torch.ones(1).to(device))
        for _ in range(n_losses)
    ])

    weights_by_key = {}
    replaced_losses = []
    original_weight = {}
    
    # Replace weights with updatable parameter
    for weight, loss_tuple in zip(loss_weights, train_closure.losses):
        _loss_tuple = list(loss_tuple)
        original_weight[_loss_tuple[-1]] = loss_tuple[1]
        _loss_tuple[1] = weight
        replaced_losses.append(tuple(_loss_tuple))
        weights_by_key[_loss_tuple[-1]] = weight
    
    sorted_keys = sorted(list(weights_by_key.keys()))
    train_closure.losses = replaced_losses

    # Setting up for least squares problem
    off_diagonal = np.eye(n_losses) - 1
    diagonal = (n_losses - 1) * np.eye(n_losses)
    
    A = off_diagonal + diagonal
    B = np.zeros(1 + n_losses)
    B[-1] = 1

    ratios = np.array([
        original_weight[key] for key in sorted_keys
    ])
    W = 1 / ratios

    loss_history = {
        key: [] for key in sorted_keys
    }

    @engine.on(ignite.engine.Events.ITERATION_COMPLETED)
    def auto_balance_weights(engine):
        if engine.state.iteration % update_frequency == 0:
            L = []
            for key in sorted_keys:
                val = weights_by_key[key]
                loss_key = f'weight/{key}'
                if loss_key not in engine.state.iter_history:
                    engine.state.iter_history[loss_key] = []
                engine.state.iter_history[loss_key].append(val.item())

                loss_history[key].append(engine.state.output[key])
            
                L.append(
                    np.percentile(
                        loss_history[key],
                        ref_percentile)
                )
            
            L = np.array(L)
            _A = A * L * W
            _A = np.vstack([_A, np.ones(n_losses)])

            # Solve with least squares for weights so each
            # loss function matches what is given in ratios.
            X = np.linalg.lstsq(_A, B, rcond=None)[0]

            # Set the weights appropriately
            for i, key in enumerate(sorted_keys):
                weights_by_key[key].data[0] = X[i]

@gin.configurable
def add_autoclip_gradient_handler(engine, model, clip_percentile):
    # Keep track of the history of gradients and select a cutoff
    # to clip values to based on percentile.
    grad_history = []

    @engine.on(nussl.ml.train.BackwardsEvents.BACKWARDS_COMPLETED)
    def autoclip_gradient(engine):
        # ignore some iterations as the grads are not useful yet
        obs_grad_norm = _get_grad_norm(model)
        grad_history.append(obs_grad_norm)

        clip_value = np.percentile(grad_history, clip_percentile)

        if 'grad_clip' not in engine.state.iter_history:
            engine.state.iter_history['grad_clip'] = []
        if 'grad_norm' not in engine.state.iter_history:
            engine.state.iter_history['grad_norm'] = []
        
        engine.state.iter_history['grad_clip'].append(clip_value)
        engine.state.iter_history['grad_norm'].append(obs_grad_norm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

class ExponentialMovingAverage:
    def __init__(self, r_value, scale=1.0):
        self.r_value = r_value
        self.scale = scale
        self.iter = 0
        self.prev_obs = 0
        
    def __call__(self, obs):
        r_value = self.r_value
        if self.iter < 1 / r_value:
            r_value = 1 / (self.iter + 1)
        est = r_value * obs + (1 - r_value) * self.prev_obs
        
        self.prev_obs = est
        self.iter += 1
        return self.scale * est
    
class ExponentialMovingPercentile:
    """
    Drawn from: here https://mjambon.com/2016-07-23-moving-percentile/.
    """
    def __init__(self, r_value, percentile, ema_r_value=None):
        self.r_value = r_value
        self.percentile = percentile
        self.iter = 0
        self.prev_obs = None
        if ema_r_value is None:
            ema_r_value = r_value
        self.ema = ExponentialMovingAverage(ema_r_value)
        
    def __call__(self, obs):
        if self.prev_obs is None:
            self.prev_obs = obs
            self.mean = self.ema(obs)
        
        est_mean = self.ema(obs)
        est_std = ((est_mean - obs) ** 2) ** .5
                
        r_value =  self.r_value
        delta = est_std * r_value 
         
        direction = obs - self.prev_obs
        
        step = 0
        if direction < 0:
            step = - delta / self.percentile
        elif direction > 0:
            step = delta / (1 - self.percentile)
                
        est = self.prev_obs + step
        self.prev_obs = est
        self.iter += 1
        
        return est