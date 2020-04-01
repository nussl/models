import nussl
import termtables
import os
import glob
import gin
import numpy as np

@gin.configurable
def analyze(output_folder):
    results_folder = os.path.join(output_folder, 'results')
    json_files = glob.glob(f"{results_folder}/*.json")
    df = nussl.evaluation.aggregate_score_files(json_files)

    overall = df.mean()
    headers = ["", f"OVERALL (N = {df.shape[0]})", ""]
    metrics = ["SAR", "SDR", "SIR"]
    data = np.array(df.mean()).T

    data = [metrics, data]
    termtables.print(data, header=headers, padding=(0, 1), alignment="ccc")

    with open(os.path.join(output_folder, 'report_card.txt'), 'w') as f:
        table = termtables.to_string(
            data, header=headers, padding=(0, 1), alignment="ccc")
        f.write(table)
        