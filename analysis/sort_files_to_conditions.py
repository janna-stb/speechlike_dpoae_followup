"""Collect the per-participant result files of a dataset and group them by condition.

For every dataset listed in `DATASETS`, the single-trial and per-participant grand-average
result files (cross-correlation, MSC, PSD) written by the `DPOAEProcessor` classes are
assigned to their experimental condition (e.g. 'Low_res F Attended') and saved as one dict
per evaluation folder. These dicts are the input of `DPOAEStatistics`.

Run from the command line with the keys of the datasets to process, e.g.::

    python sort_files_to_conditions.py js25 js26 msa21
"""

import sys
from pathlib import Path

import numpy as np

# Datasets processed when no command-line arguments are given (keys of DATASETS).
RUN_DATASETS = ['js25', 'js26', 'msa21']

# Root folder that contains one folder per dataset (see README). Adjust to your local copy.
DATA_ROOT = Path('path/to/data')


def _subject_ids(prefix, start, stop):
    """Return participant IDs `prefix` + zero-padded number for numbers in [start, stop)."""
    return [f'{prefix}{str(p).zfill(2)}' for p in range(start, stop)]


def _excluding(subjects, exclude):
    """Return `subjects` without the IDs listed in `exclude`."""
    return [s for s in subjects if s not in exclude]


# dataset: key of CONDITION_EXTRACTORS; data_path: dataset folder; subjects: participants included;
# evaluation_folders: result folders written by the processor classes
DATASETS = {
    'js25': dict(
        dataset='JS25',
        data_path=DATA_ROOT / 'JS25',
        subjects=_excluding(_subject_ids('P', 1, 41), ['P13', 'P20']),
        evaluation_folders=['Competing_Speaker'],
    ),
    'js26': dict(
        dataset='JS26',
        data_path=DATA_ROOT / 'JS26',
        subjects=_subject_ids('P', 1, 45),
        evaluation_folders=['DPOAE_analysis_contra'],
    ),
    'msa21': dict(
        dataset='MSA21',
        data_path=DATA_ROOT / 'MSA21',
        subjects=_excluding(_subject_ids('P', 11, 14) + ['P16'] + _subject_ids('P', 18, 41), ['P24', 'P29', 'P39']),
        evaluation_folders=['DPOAE_analysis'],
    ),
    'js25_outofear': dict(
        dataset='JS25',
        data_path=DATA_ROOT / 'JS25_outofear',
        subjects=_subject_ids('outofear_', 1, 13),
        evaluation_folders=['Competing_Speaker'],
    ),
    'js26_outofear': dict(
        dataset='JS26',
        data_path=DATA_ROOT / 'JS26',  # in-ear and out-of-ear recordings share one dataset folder
        subjects=_subject_ids('outofear_', 1, 13),
        evaluation_folders=['DPOAE_analysis_contra'],
    ),
    'msa21_outofear': dict(
        dataset='MSA21',
        data_path=DATA_ROOT / 'MSA21_outofear',
        subjects=_subject_ids('outofear_', 1, 13),
        evaluation_folders=['DPOAE_analysis'],
    ),
}


def main(datasets=None):
    """Sort the result files of the selected datasets, once for each evaluation folder.

    Parameters
    ----------
    datasets : list of str, optional
        Keys of `DATASETS`. Defaults to the command-line arguments if given, else `RUN_DATASETS`.
    """
    if datasets is None:
        datasets = sys.argv[1:] or RUN_DATASETS

    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        raise ValueError(f'Unknown dataset(s) {unknown}. Choose from {list(DATASETS)}')

    for name in datasets:
        cfg = DATASETS[name]
        print(f'--- {name} ---')
        for subfolder in cfg['evaluation_folders']:
            sort_files_to_conditions(cfg['subjects'], cfg['data_path'], subfolder, dataset=cfg['dataset'])


def sort_files_to_conditions(participants, data_path, subfolder_name, dataset='JS26'):
    """Group the result files of all participants of one dataset by condition and save them.

    Files are read from `data_path/results_<participant>/<subfolder_name>/Successful_attention`.
    Single-trial and grand-average files of the three methods are matched by their sorted
    order, and each file is assigned to a condition from its folder structure.

    Parameters
    ----------
    participants : list of str
        Participant IDs, e.g. 'P01' or 'outofear_01'.
    data_path : str or Path
        Dataset folder containing the `results_<participant>` folders.
    subfolder_name : str
        Evaluation folder within each participant's results, e.g. 'DPOAE_analysis_contra'.
    dataset : str, optional
        'JS25', 'JS26' or 'MSA21'; selects the condition extractor. Default is 'JS26'.

    Notes
    -----
    Saves a dict {condition: {'crosscorr': [...], 'msc': [...], 'psd': [...],
    'GA_crosscorr': [...], 'GA_msc': [...], 'GA_psd': [...]}} to
    `data_path/Statistical_Analysis/<first>_to_<last>/<subfolder_name>_files_per_condition.npy`
    (for out-of-ear recordings directly to `data_path/Statistical_Analysis/`).
    """
    data_path = Path(data_path)
    if participants[0].startswith('outofear'):
        res_path = data_path / 'Statistical_Analysis'
    else:
        res_path = data_path / f'Statistical_Analysis/{participants[0]}_to_{participants[-1]}'
    res_path.mkdir(parents=True, exist_ok=True)

    condition_extractor = get_condition_extractor(dataset)
    cond_files = {}

    for p in participants:
        p_path = data_path / f'results_{p}/{subfolder_name}/Successful_attention'
        all_files_crosscorr = sorted(p_path.rglob('*corr.npy'))
        all_files_msc = sorted(p_path.rglob('*MSC_results.npy'))
        all_files_psd = sorted(p_path.rglob('*PSD_results.npy'))
        ga_files_crosscorr = sorted(p_path.rglob('GA_data.npy'))
        ga_files_msc = sorted(p_path.rglob('GA_MSC_data.npy'))
        ga_files_psd = sorted(p_path.rglob('GA_PSD_data.npy'))

        assert len(all_files_crosscorr) == len(all_files_msc) == len(all_files_psd), \
            f'Mismatch in number of files for participant {p} in {subfolder_name}'
        assert len(ga_files_crosscorr) == len(ga_files_msc) == len(ga_files_psd), \
            f'Mismatch in number of GA files for participant {p} in {subfolder_name}'

        for file_crosscorr, file_msc, file_psd in zip(all_files_crosscorr, all_files_msc, all_files_psd):
            cond = cond_files.setdefault(condition_extractor(file_crosscorr),
                                         {'crosscorr': [], 'msc': [], 'psd': [], 'GA_crosscorr': [], 'GA_msc': [], 'GA_psd': []})
            cond['crosscorr'].append(file_crosscorr)
            cond['msc'].append(file_msc)
            cond['psd'].append(file_psd)

        for ga_crosscorr, ga_msc, ga_psd in zip(ga_files_crosscorr, ga_files_msc, ga_files_psd):
            cond = cond_files[condition_extractor(ga_crosscorr)]
            cond['GA_crosscorr'].append(ga_crosscorr)
            cond['GA_msc'].append(ga_msc)
            cond['GA_psd'].append(ga_psd)

    np.save(res_path / f'{subfolder_name}_files_per_condition.npy', cond_files)


def _voice_and_attention(target_folder, dp_folder):
    """Map the target folder and the Attended/Ignored DP folder to the condition suffix.

    The DP waveform of the attended voice belongs to the target voice, that of the ignored
    voice to the other voice, e.g. ('Target_male', 'Ignored_DPs') -> 'F Ignored'.
    """
    mapping = {('Target_male', 'Attended_DPs'): 'M Attended', ('Target_male', 'Ignored_DPs'): 'F Ignored',
               ('Target_female', 'Attended_DPs'): 'F Attended', ('Target_female', 'Ignored_DPs'): 'M Ignored'}
    return mapping[(target_folder, dp_folder)]


def extract_condition_from_filepath_JS26(filepath):
    """Derive the JS26 condition name from the path of a result file.

    Parameters
    ----------
    filepath : str or Path
        Path of the form `.../Target_<male|female>/<harm_cond>/<Attended_DPs|Ignored_DPs>/<file>`.

    Returns
    -------
    str
        Condition name, e.g. 'Low_res M Attended'.
    """
    parts = Path(filepath).parts
    return f'{parts[-3].capitalize()} {_voice_and_attention(parts[-4].capitalize(), parts[-2])}'


def extract_condition_from_filepath_JS25(filepath):
    """Derive the JS25 condition name from the path of a result file.

    Parameters
    ----------
    filepath : str or Path
        Path of the form `.../Target_<target>/<resolved|unresolved>/<Attended_DPs|Ignored_DPs>/<file>`.
        Baseline grand averages combined by `DPOAEProcessorJS25.average_baseline_GAs` lie
        directly in `.../Target_video_BL_Polarnacht_<voice>/<resolved|unresolved>/`.

    Returns
    -------
    str
        Condition name, e.g. 'Resolved F Attended' or 'Unresolved M Baseline_DPs'.
    """
    parts = Path(filepath).parts
    t = next(i for i in range(len(parts) - 1, -1, -1) if parts[i].startswith('Target_'))
    target_folder, harm_type, last_folder = parts[t], parts[t + 1], parts[-2]

    if target_folder in ('Target_male', 'Target_female'):
        suffix = _voice_and_attention(target_folder, last_folder)
    else:
        # baseline: Polarnacht is read by the voice in the folder name, Darum by the other voice
        bl_voice = 'F' if target_folder.endswith('female') else 'M'
        other_voice = 'M' if bl_voice == 'F' else 'F'
        suffix = f'{other_voice if last_folder == "Ignored_DPs" else bl_voice} Baseline_DPs'

    return f'{harm_type.capitalize()} {suffix}'


def extract_condition_from_filepath_MSA21(filepath):
    """Derive the MSA21 condition name from the path of a result file.

    Parameters
    ----------
    filepath : str or Path
        Path of the form `.../<Attended|Ignored>_<female|male>/<file>`.

    Returns
    -------
    str
        Condition name, e.g. 'F Attended'.
    """
    attention, voice = Path(filepath).parts[-2].split('_')
    return f"{'F' if voice == 'female' else 'M'} {attention}"


CONDITION_EXTRACTORS = {'JS25': extract_condition_from_filepath_JS25,
                        'JS26': extract_condition_from_filepath_JS26,
                        'MSA21': extract_condition_from_filepath_MSA21}


def get_condition_extractor(dataset):
    """Return the condition extractor of a dataset.

    Parameters
    ----------
    dataset : str
        'JS25', 'JS26' or 'MSA21'.

    Returns
    -------
    callable
        Function `(filepath) -> condition name`.
    """
    try:
        return CONDITION_EXTRACTORS[dataset]
    except KeyError as exc:
        raise ValueError(f'Unsupported dataset: {dataset}. Choose from {list(CONDITION_EXTRACTORS)}') from exc


if __name__ == '__main__':
    main()
