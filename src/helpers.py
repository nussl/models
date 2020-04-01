import gin
import logging
import os

logging.basicConfig(	
    format='%(asctime)s,%(msecs)d [%(filename)s:%(lineno)d] %(message)s',	
    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO)

@gin.configurable
def output_folder(_output_folder):
    return _output_folder

@gin.configurable
def model_path(model_suffix):
    _output_folder = output_folder()
    _model_path = os.path.join(_output_folder, model_suffix)
    return _model_path

@gin.configurable
def build_dataset(dataset_class):
    return dataset_class()
