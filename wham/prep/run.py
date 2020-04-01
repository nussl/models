#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import os
import argparse

def run(cmd):
    print(cmd)
    subprocess.run([cmd], shell=True)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_directory', 
        default=os.getenv('DATA_DIRECTORY'), type=str,
        help="Everything will get downloaded and built into this folder")

    WSJ_DIRECTORY = os.path.join(
        os.getenv('DATA_DIRECTORY'), 'wsj0')
    
    parser.add_argument('-w', '--wsj_directory', 
        default=WSJ_DIRECTORY, type=str,
        help="Location of WSJ0.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()

    run(f"curl https://storage.googleapis.com/whisper-public/wham_noise.zip "
        f"-o ${args.data_directory}/wham_noise.zip")

    os.makedirs(os.path.join(
        args.data_directory, 'wham'), 
        exist_ok=True)
    
    run(f"unzip {args.data_directory}/wham_noise.zip "
        f"-d {args.data_directory}/wham/wham-noise")
    
    run(f"cd scripts && python create_wham_from_scratch.py " 
        f"--wsj0-root {args.wsj_directory} "
        f"--wham-noise-root {args.data_directory}/wham/wham_noise "
        f"--output-dir {args.data_directory}/wham/ "
    )
