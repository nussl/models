import gin
import argparse
from src import (
    train, 
    cache, 
    analyze, 
    instantiate, 
    evaluate, 
    helpers, 
    mix_with_scaper,
    make_scaper_datasets
)
import nussl
import subprocess
from src.helpers import build_logger
from src.debug import DebugDataset
import os
import copy

def edit(experiment_config):
    subprocess.run([
        f'vim {experiment_config}'
    ], shell=True)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('func', type=str)
    parser.add_argument('-exp', '--experiment_config', type=str)
    parser.add_argument('-dat', '--data_config', default=None, type=str)
    parser.add_argument('-env', '--environment_config', default=None, type=str)
    parser.add_argument('-out', '--output_folder', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    special_commands = ['all', 'debug']

    if args.func not in special_commands:
        if args.func not in globals():
            raise ValueError(f"No matching function named {args.func}!")
        func = globals()[args.func]

    _configs = [
        args.environment_config, 
        args.data_config, 
        args.experiment_config
    ]

    for _config in _configs:
        if _config is not None:
            gin.parse_config_file(_config)

    build_logger()

    if args.func == 'debug':
        # overfit to a single batch for a given length.
        # save the model
        # evaluate it on that same sample
        # do this via binding parameters to gin config
        # then set args.func = 'all'
        debug_output_folder = os.path.join(
            helpers.output_folder(), 'debug')
        gin.bind_parameter(
            'output_folder._output_folder', 
            debug_output_folder
        )
        with gin.config_scope('train'):
            train_dataset = helpers.build_dataset()
        
        test_dataset = copy.deepcopy(train_dataset)
        test_dataset.transform = None
        test_dataset.cache_populated = False

        train_dataset = DebugDataset(train_dataset)
        val_dataset = copy.deepcopy(train_dataset)
        val_dataset.dataset_length = 1

        test_dataset = DebugDataset(test_dataset)
        test_dataset.dataset_length = 1
        test_dataset.idx = train_dataset.idx

        gin.bind_parameter('train/build_dataset.dataset_class', train_dataset)
        gin.bind_parameter('val/build_dataset.dataset_class', val_dataset)
        gin.bind_parameter('test/build_dataset.dataset_class', test_dataset)

        gin.bind_parameter('train.num_epochs', 1)

        args.func = 'all'

    if args.func == 'all':
        train()
        evaluate()
        analyze()
    elif args.func == 'instantiate':
        func(args.output_folder)
    elif args.func == 'edit':
        func(args.experiment_config)
    elif args.func == 'cache':
        def _setup_for_cache(scope):
            with gin.config_scope(scope):
                _dataset = helpers.build_dataset()
                _dataset.cache_populated = False
                gin.bind_parameter(
                    f'{scope}/build_dataset.dataset_class', 
                    _dataset
                )
        for scope in ['train', 'val', 'test']:
            _setup_for_cache(scope)
        cache()
    else:
        func()
