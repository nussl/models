import nussl
import os
import glob
import gin

@gin.configurable
def analyze(output_folder, notes=None):
    results_folder = os.path.join(output_folder, 'results')
    json_files = glob.glob(f"{results_folder}/*.json")

    df = nussl.evaluation.aggregate_score_files(json_files)
    df['source'] = df['source'].apply(lambda x: x.split('_')[-1][:-4])
    report_card = nussl.evaluation.report_card(df, notes=notes)
    print(report_card)
    
    with open(os.path.join(output_folder, 'report_card.txt'), 'w') as f:
        f.write(report_card)
