import nussl
import glob
import os
from argparse import ArgumentParser
import tqdm
import p_tqdm

def _convert(dataset, i, output_directory, folder_name):
    name = dataset.musdb[i].name
    item = dataset[i]

    for key, signal in item['sources'].items():
        output_path = os.path.join(output_directory, folder_name, key)
        os.makedirs(output_path, exist_ok=True)
        signal.write_audio_to_file(
            os.path.join(output_path, f'{name}.wav'))

def convert(input_directory, output_directory):
    subsets = ['train', 'test']

    dataset_args = [
        {'subsets': ['train'], 'split': 'train', 'folder_name': 'train'},
        {'subsets': ['train'], 'split': 'valid', 'folder_name': 'valid'},
        {'subsets': ['test'], 'folder_name': 'test'},
    ]

    for dataset_arg in dataset_args:
        folder_name = dataset_arg.pop('folder_name')
        dataset = nussl.datasets.MUSDB18(input_directory, **dataset_arg)
        indices = list(range(len(dataset)))

        args = [
            [dataset for _ in indices],
            indices,
            [output_directory for _ in indices],
            [folder_name for _ in indices]
        ]

        p_tqdm.p_map(_convert, *args, num_cpus=10)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('input_directory', type=str, 
        help="Where the MUSDB data in stempeg is.")
    parser.add_argument('output_directory', type=str, 
        help="Where each track should be saved.")
    args = parser.parse_args()
    args = vars(args)
    convert(args['input_directory'], args['output_directory'])
