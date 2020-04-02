import nussl
import os
import glob
import gin

@gin.configurable
def analyze(output_folder, notes=None, report_each_source=True):
    results_folder = os.path.join(output_folder, 'results')
    json_files = glob.glob(f"{results_folder}/*.json")

    df = nussl.evaluation.aggregate_score_files(json_files)
    report_card = nussl.evaluation.report_card(
        df, notes=notes, report_each_source=report_each_source)

    print(report_card)
    with open(os.path.join(output_folder, 'report_card.txt'), 'w') as f:
        f.write(report_card)
