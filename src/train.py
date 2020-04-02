from .helpers import build_dataset
import nussl
import gin
import torch
import os
import logging
from torch import multiprocessing
from ignite.contrib.handlers import ProgressBar

@gin.configurable
def add_train_handlers(engine, model, scheduler, handler_names):
    for handler_name in handler_names:
        if handler_name == "add_clip_gradient_handler":
            add_clip_gradient_handler(engine, model)
        elif handler_name == "add_lr_scheduler_handler":
            add_lr_scheduler_handler(engine, scheduler)

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
def build_model_optimizer_scheduler(model_config, optimizer_class, 
                                    scheduler_class, device='cuda'):
    model = nussl.ml.SeparationModel(model_config).to(device)
    # the rest of optimizer params comes from gin
    optimizer = optimizer_class(model.parameters())
    scheduler = scheduler_class(optimizer)
    return model, optimizer, scheduler

@gin.configurable
def train(batch_size, loss_dictionary, num_data_workers, seed,
          output_folder, num_epochs, device='cuda'):
    with gin.config_scope('train'):
        train_dataset = build_dataset()
    with gin.config_scope('val'):
        val_dataset = build_dataset()

    model, optimizer, scheduler = build_model_optimizer_scheduler(device=device)
    logging.info(model)
    
    if not torch.cuda.is_available():
        device = 'cpu'

    os.makedirs(output_folder, exist_ok=True)
    logging.info(f'Saving to {output_folder}')

    nussl.utils.seed(seed)
    num_data_workers = min(num_data_workers, multiprocessing.cpu_count())

    # Set up dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, num_workers=num_data_workers, 
        batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, num_workers=num_data_workers, 
        batch_size=batch_size, shuffle=True)

    # Build the closures for each loop
    train_closure = nussl.ml.train.closures.TrainClosure(
        loss_dictionary, optimizer, model)
    val_closure = nussl.ml.train.closures.ValidationClosure(
        loss_dictionary, model)

    # Build the engine and add handlers
    train_engine, val_engine = nussl.ml.train.create_train_and_validation_engines(
        train_closure, val_closure, device=device)
    nussl.ml.train.add_validate_and_checkpoint(
        output_folder, model, optimizer, train_dataset, train_engine,
        val_data=val_dataloader, validator=val_engine)
    nussl.ml.train.add_stdout_handler(train_engine, val_engine)
    nussl.ml.train.add_tensorboard_handler(output_folder, train_engine)
    nussl.ml.train.add_progress_bar_handler(train_engine, val_engine)

    # clip_value and scheduler come from gin config
    add_train_handlers(
        engine, model, scheduler, 
        [
            'add_clip_gradient_handler',
            'add_lr_scheduler_handler'
        ]
    )
    
    # run the engine
    train_engine.run(train_dataloader, max_epochs=num_epochs)

@gin.configurable
def cache(num_cache_workers, batch_size):
    num_cache_workers = min(
        num_cache_workers, multiprocessing.cpu_count())
    datasets = []
    for scope in ['train', 'val']:
        with gin.config_scope(scope):
            dataset = build_dataset()
            cache_dataloader = torch.utils.data.DataLoader(
                dataset, num_workers=num_cache_workers, 
                batch_size=batch_size)
            nussl.ml.train.cache_dataset(cache_dataloader)
    
    alert = "Make sure to change cache_populated = True in your gin config!"
    border = ''.join(['=' for _ in alert])

    logging.info(
        f'\n\n{border}\n'
        f'{alert}\n'
        f'{border}\n'
    )
