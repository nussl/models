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
    parser.add_argument('-d', '--musdb_directory',  type=str,
        help="Location of unzipped MUSDB directory with stems.")

    parser.add_argument('-o', '--output_directory', 
        type=str, help="Where to put processed MUSDB.")

    parser.add_argument('-s', '--sample_rate', default=16000,
        type=int, help="Sample rate to resample each file to.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()

    os.makedirs(os.path.join(
        args.output_directory, 'scaper'), 
        exist_ok=True)
    
    run(f"python convert.py " 
        f"{args.musdb_directory} "
        f"{args.output_directory} "
    )

    resample_directory = os.path.join(
        args.output_directory, f'wav{args.sample_rate}/')

    run(f"python resample.py "
        f"--input_path {args.output_directory} "
        f"--output_path {resample_directory} "
        f"--sample_rate {args.sample_rate} "
        f"--num_workers 10 "
    )
