import gin
import argparse
from src import train, cache, analyze, instantiate, evaluate

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('func', type=str)
    parser.add_argument('-exp', '--experiment_config', type=str)
    parser.add_argument('-dat', '--data_config', default=None, type=str)
    parser.add_argument('-env', '--environment_config', default=None, type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()

    if args.func != 'all':
        if args.func not in globals():
            raise ValueError(f"No matching function named {args.func}!")
        func = globals()[args.func]

    _configs = [args.environment_config, args.data_config, args.experiment_config]

    for _config in _configs:
        if _config is not None:
            gin.parse_config_file(_config)

    if args.func == 'all':
        train()
        evaluate()
        analyze()
    elif args.func == 'instantiate':
        func(args.experiment_config)
    else:
        func()
