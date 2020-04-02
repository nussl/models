import itertools
import os
import gin
import logging

@gin.configurable
def sweep(parameters):
    keys = sorted(list(parameters.keys()))
    values = [parameters[k] for k in keys]
    cartesian_product = itertools.product(*values)
    sweep_as_dict = {}

    for setting in cartesian_product:
        for i, v in enumerate(setting):
            gin.bind_parameter(keys[i], v)
            sweep_as_dict[keys[i]] = v
        sweep_as_str = [
            f"{k.split('.')[-1]}:{v}"
            for k, v in sweep_as_dict.items()
        ]
        sweep_as_str = '-'.join(sweep_as_str)
        yield sweep_as_str

def instantiate(folder):
    os.makedirs(folder, exist_ok=True)

    def get_run_number(path):
        return len([
            x for x in os.listdir(path) 
        ])

    def write_gin_config(output_folder, swp=''):
        if swp is not '':
            output_folder += f':{swp}'
        os.makedirs(output_folder, exist_ok=True)
        gin.bind_parameter(
            'output_folder._output_folder', 
            os.path.abspath(output_folder)
        )
        output_path = os.path.join(output_folder, 'config.gin')
        with open(output_path, 'w') as f:
            logging.info(f'{swp} -> {output_path}')
            f.write(gin.config_str())

    run_number = get_run_number(folder) 

    try:
        for i, swp in enumerate(sweep()):
            output_folder = os.path.join(
                folder, f'run{run_number + i}')
            write_gin_config(output_folder, swp)
    except:
        output_folder = os.path.join(
            folder, f'run{run_number}')
        write_gin_config(output_folder)
