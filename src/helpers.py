import gin
import logging
import os
from datetime import datetime
import sys
import nussl

@gin.configurable
def join_path(base_path, relative_path):
    return os.path.join(base_path, relative_path)

@gin.configurable
def output_folder(_output_folder=None):
    return _output_folder

@gin.configurable
def model_path(model_suffix):
    _output_folder = output_folder()
    _model_path = os.path.join(_output_folder, model_suffix)
    return _model_path

@gin.configurable
def build_dataset(dataset_class):
    if isinstance(dataset_class, type):
        # Not instantiated yet
        return dataset_class()
    else:
        # Already instantiated
        return dataset_class

@gin.configurable
def build_transforms(transform_names_and_args, cache_location):
    tfms = []
    for tfm_name, tfm_args in transform_names_and_args:
        if tfm_name == 'Cache':
            tfm_args['location'] = cache_location
        tfm = getattr(nussl.datasets.transforms, tfm_name)
        tfms.append(tfm(**tfm_args))
    return nussl.datasets.transforms.Compose(tfms)

def build_logger():
    _output_folder = output_folder()
    now = datetime.now()
    if _output_folder is not None:
        logging_file = os.path.join(
            _output_folder, 'logs', now.strftime("%Y.%m.%d-%H.%M.%S") + '.log')

        os.makedirs(os.path.join(_output_folder, 'logs'), exist_ok=True)
            
        logging.basicConfig(
            level = logging.INFO,
            format='%(asctime)s,%(msecs)d [%(filename)s:%(lineno)d] %(message)s',
            handlers=[
                logging.FileHandler(logging_file),
                logging.StreamHandler(sys.stdout),
            ]
        )
    else:
        logging.basicConfig(
            level = logging.INFO,
            format='%(asctime)s,%(msecs)d [%(filename)s:%(lineno)d] %(message)s',
        )
