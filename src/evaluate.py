import nussl
import os
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import tqdm
import gin
from .helpers import build_dataset
import logging

@gin.configurable
def evaluate(output_folder, separation_algorithm, eval_class, 
             block_on_gpu, num_workers, seed):
    nussl.utils.seed(seed)
    logging.info(gin.operative_config_str())
    
    with gin.config_scope('test'):
        test_dataset = build_dataset()
    
    results_folder = os.path.join(output_folder, 'results')
    os.makedirs(results_folder, exist_ok=True)
    set_model_to_none = False

    if block_on_gpu:
        # make an instance that'll be used on GPU
        # has an empty audio signal for now
        gpu_algorithm = separation_algorithm(
            nussl.AudioSignal(), device='cuda')
        set_model_to_none = True

    def forward_on_gpu(audio_signal):
        # set the audio signal of the object to this item's mix
        gpu_algorithm.audio_signal = audio_signal
        if hasattr(gpu_algorithm, 'forward'):
            gpu_output = gpu_algorithm.forward()
        elif hasattr(gpu_algorithm, 'extract_features'):
            gpu_output = gpu_algorithm.extract_features()
        return gpu_output

    pbar = tqdm.tqdm(total=len(test_dataset))

    def separate_and_evaluate(item, gpu_output):
        if set_model_to_none:
            separator = separation_algorithm(item['mix'], model_path=None)
        else:
            separator = separation_algorithm(item['mix'])
        estimates = separator(gpu_output)
        source_names = sorted(list(item['sources'].keys()))
        sources = [item['sources'][k] for k in source_names]
        
        # other arguments come from gin config
        evaluator = eval_class(sources, estimates)
        scores = evaluator.evaluate()
        output_path = os.path.join(
            results_folder, f"{item['mix'].file_name}.json")
        with open(output_path, 'w') as f:
            json.dump(scores, f, indent=2)
        pbar.update(1)
    
    pool = ThreadPoolExecutor(max_workers=num_workers)
    
    for i in range(len(test_dataset)):
        item = test_dataset[i]
        gpu_output = forward_on_gpu(item['mix'])
        if i == 0:
            separate_and_evaluate(item, gpu_output)
            continue
        pool.submit(separate_and_evaluate, item, gpu_output)
    
    pool.shutdown(wait=True)
