import nussl
import termtables
import os
import glob
import gin
import numpy as np

def _get_mean_and_std(df):
    excluded_columns = ['source', 'file']

    metrics = [x for x in list(df.columns) if x not in excluded_columns]

    means = [f'{m:.03f}' for m in np.array(df.mean()).T]
    stds = [f'{s:.03f}' for s in np.array(df.std()).T]
    data = [f'{m} +/- {s}' for m, s in zip(means, stds)]

    return metrics, data

def _get_medians(df):
    excluded_columns = ['source', 'file']

    metrics = [x for x in list(df.columns) if x not in excluded_columns]

    # this strange padding is so the reports look nice when printed back to back
    data = [f'     {m:.03f}     ' for m in np.array(df.median()).T]

    return metrics, data

def _get_report_card(df, func, func_name=''):
    labels, data = func(df)
    labels.insert(0, func_name)
    data.insert(0, 'OVERALL')
    data = [data]

    for name in np.unique(df['source']):
        _df = df[df['source'] == name]
        _, _data = func(_df)
        _data.insert(0, name.upper())
        data.append(_data)
        
    report_card = termtables.to_string(
        data, header=labels, padding=(0, 1), alignment="cccc")

    return report_card

@gin.configurable
def analyze(output_folder, notes=None):
    results_folder = os.path.join(output_folder, 'results')
    json_files = glob.glob(f"{results_folder}/*.json")
    df = nussl.evaluation.aggregate_score_files(json_files)
    df.reset_index(inplace=True) 

    df['source'] = df['source'].apply(lambda x: x.split('_')[-1][:-4])

    mean_report_card = _get_report_card(df, _get_mean_and_std)
    median_report_card = _get_report_card(df, _get_medians)

    line_break = mean_report_card.index('\n')

    def _format_title(title, marker="-"):
        pad = (line_break - len(title)) // 2
        pad = ''.join([marker for _ in range(pad)])
        border = pad + title + pad
        if len(title) % 2:
            border = border + marker
        return border

    report_card = (
        f"{_format_title('')}\n"
        f"{_format_title(' MEAN +/- STD OF METRICS ')}\n"
        f"{_format_title('')}\n"
        f"{mean_report_card}\n"
        f"{_format_title('')}\n"
        f"{_format_title(' MEDIAN OF METRICS ')}\n"
        f"{_format_title('')}\n"
        f"{median_report_card}\n"
    )
    
    if notes is not None:
        report_card += (
            f"{_format_title('')}\n"
            f"{_format_title(' NOTES ')}\n"
            f"{_format_title('')}\n"
            f"{notes}"
        )

    print(report_card)
    
    with open(os.path.join(output_folder, 'report_card.txt'), 'w') as f:
        f.write(report_card)
