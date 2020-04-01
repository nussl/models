from .helpers import build_dataset
import nussl
import gin
import torch
import os
import logging
import multiprocessing

@gin.configurable
def add_clip_gradient_handler(engine, model, clip_value):
    @engine.on(nussl.ml.train.BackwardsEvents.BACKWARDS_COMPLETED)
    def clip_gradient(engine):
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

def add_lr_scheduler_handler(engine, scheduler):
    @engine.on(nussl.ml.train.ValidationEvents.VALIDATION_COMPLETED)
    def step_scheduler(trainer):
        val_loss = engine.state.epoch_history['validation/loss'][-1]
        scheduler.step(val_loss)

@gin.configurable
def build_model_optimizer_scheduler(model_config, optimizer_class, 
                                    scheduler_class):
    model = nussl.ml.SeparationModel(model_config)
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

    model, optimizer, scheduler = build_model_optimizer_scheduler()
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
        batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, num_workers=num_data_workers, 
        batch_size=batch_size)

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

    # clip_value and scheduler come from gin config
    add_clip_gradient_handler(train_engine, model)
    add_lr_scheduler_handler(train_engine, scheduler)

    # run the engine
    train_engine.run(train_dataloader)

def cache():
    datasets = []
    for scope in ['train', 'val']:
        with gin.config_scope(scope):
            dataset = build_dataset()
            nussl.ml.train.cache_dataset(dataset)
