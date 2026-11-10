# speechlike_dpoae_followup

Analysis and stimulus-generation code for measuring speech-like distortion product otoacoustic emissions (DPOAEs) with speech-derived stimuli. The code accompanies two manuscripts (in preparation):

- a **methods paper** comparing metrics for detecting speech-like DPOAEs (cross-correlation, power spectral density, magnitude-squared coherence) across three datasets, and
- a paper reporting the outcome of a **second experiment on attentional effects on speech-like DPOAEs** in a competing-speaker paradigm with pitch-shifted voices, which could not reproduce the attentional effects reported in Steinebach & Reichenbach (2026) (see below).

The repository contains the code to generate the stimuli of the second experiment, the processing of the recordings, and all subsequent analyses of the derived metrics.

## Relation to the earlier repository

The code for Steinebach & Reichenbach (2026), *Frontiers in Neuroscience*, [doi:10.3389/fnins.2026.1756386](https://doi.org/10.3389/fnins.2026.1756386), is available at [janna-stb/speechlike_dpoae_FrontNeurosci](https://github.com/janna-stb/speechlike_dpoae_FrontNeurosci) and is kept unchanged as the record of that publication.

This repository is a further development of that code, not a drop-in replacement:

- The processing is restructured into a base class (`DPOAEProcessor`) with small dataset-specific subclasses.
- Power spectral density (PSD) and magnitude-squared coherence (MSC) were added as metrics.
- The statistics module adds Gaussian fits of the grand-average cross-correlations and AUC-based measures.
- Several default parameters differ, e.g. the window in which the cross-correlation peak is searched.
- The estimation of stimulus levels was corrected. The JS25 stimulus levels initially reported in the Frontiers paper were too low; they have been corrected.

Running this code on the JS25 data therefore does not reproduce the numbers of Steinebach & Reichenbach (2026) exactly. The stimulus-generation code also differs from the earlier version (f0 estimation with Praat, pitch shifting of the audiobooks, an anti-aliasing low-pass filter and a different normalisation); the JS25 stimuli are generated with the earlier repository.

## Datasets

- **MSA21**: Saiz-Alía & Reichenbach (2021), *eNeuro*, data on figshare ([doi.org/10.6084/m9.figshare.12738515](https://doi.org/10.6084/m9.figshare.12738515)). Processed with `DPOAEProcessorMSA21`.
- **JS25**: Steinebach & Reichenbach (2026), *Frontiers in Neuroscience*, data on Zenodo ([doi:10.5281/zenodo.16837673](https://doi.org/10.5281/zenodo.16837673)). Processed with `DPOAEProcessorJS25`.
- **JS26**: second experiment with pitch-shifted voices, data on Zenodo (upload pending). Processed with `DPOAEProcessor` (alias `DPOAEProcessorJS26`).

Each dataset also processed a set of out-of-ear control recordings (participants `outofear_01`, ...). For JS26 they are published alongside the participant recordings on Zenodo (`Recordings_outofear/`).

## Repository structure

- `analysis/`
    - `DPOAE_processor.py`: per-participant processing; base class, used directly for JS26
    - `DPOAE_processor_JS25.py`: subclass for the file layout of JS25
    - `DPOAE_processor_MSA21.py`: subclass for the file layout of MSA21
    - `sort_files_to_conditions.py`: collects the per-participant results per condition
    - `DPOAE_statistics.py`: grand averages and the data for the statistical comparisons
    - `DPOAE_plotting.py`: box and bar plots with significance tests
    - `functions_significance_testing.py`: statistical tests, FDR correction, significance bars
    - `plot_grand_averages.py`: peak and noise-floor estimation, grand-average plots
    - `functions_crosscorr.py`: complex cross-correlation
    - `functions_filter.py`: Butterworth filter
- `stimulus_generation/`
    - `create_noise_segments.py`: step 1, cut the competing-speaker segments
    - `generate_shifted_books_and_waveforms.py`: step 2, pitch-shift the audiobooks and create the harmonic and DP waveforms
    - `generate_multiband_stimuli.py`: step 3, combine the waveforms into the presented stimuli and mix the audiobooks
    - `class_stimulus_gen.py`: generation of harmonic waveforms and multiband stimuli
    - `functions_f0_shift.py`: pitch shifting with Praat (PSOLA)
    - `plot_stimulus_correlations.py`, `running_mean.py`: helpers
- `environment.yml`: conda environment

The stimulus-generation code uses `functions_filter.py` and `functions_crosscorr.py` from `analysis/`.

## Installation

```bash
conda env create -f environment.yml
conda activate speechlike_dpoae
```

The modules import each other by name, so run the code from within `analysis/` or `stimulus_generation/`, or add `analysis/` to your `PYTHONPATH`.

## Data layout

Paths to the data are not part of the repository. Set them where indicated: `DATA_ROOT` in `sort_files_to_conditions.py`, and `data_path` / `main_path` at the top of the stimulus scripts. Each processor class expects one folder per dataset (`exp_path`) and writes its results to `exp_path/results_<participant>/`. The expected layout is described in the docstring of each class:

- **JS26** (`DPOAEProcessor`):
    - `Recordings_inear/data_<participant>/`: `<participant>_psychopy_info.csv`, `<participant>_target_info.csv`, `<participant>_recording_OAE_PT.wav` and the folder `<participant>_recordings_contralateral/`
    - `Recordings_outofear/data_<participant>/`: same structure for the out-of-ear recordings
    - `Stimuli/`: presented stimuli (`<harm_cond>/competing_stimuli_.../`) and DP waveforms (`DP_STORY_<voice>/`, `DP_NOISE_<voice>/`, each with one subfolder per f0, `f0_to_<f0>Hz/`)
- **JS25** (`DPOAEProcessorJS25`): as published on Zenodo
    - `Recordings/data_<participant>/`: recordings and experiment files
    - `Stimuli/`: presented stimuli and DP waveforms
- **MSA21** (`DPOAEProcessorMSA21`): as in the original dataset
    - `data/<participant>_DONE/`: recordings and responses
    - `Waveforms_Female/`, `Waveforms_Male/`: harmonic and DP waveforms

## Usage

### 1. Per-participant processing

Once per dataset folder (including the out-of-ear folders), compute the autocorrelations of the DP waveforms and the DPOAE frequency bands. The autocorrelations are used for the Gaussian fits in step 3, the frequency bands for the MSC and PSD:

```python
from DPOAE_processor import DPOAEProcessor

proc = DPOAEProcessor('Autocorrelations', exp_path)   # results in exp_path/results_Autocorrelations
proc.compute_autocorrelations()
proc.estimate_DPOAE_freqbands()                       # saved as exp_path/dpoae_bandwidths.npy
```

Then process every participant, here for JS26:

```python
for sbj in participants:
    proc = DPOAEProcessor(sbj, exp_path)
    proc.analyze_PT_DPOAE()                 # pure-tone DPOAE check (JS26 only), returns True if significant
    proc.analyze_speech_DPOAE()             # recordings of the ear contralateral to the audiobook
    proc.average_crosscorr_per_condition()
    proc.average_MSC_per_condition()
    proc.average_PSD_per_condition()
```

- **JS25**: use `DPOAEProcessorJS25` and `analyze_speech_DPOAE('Competing')`, and call `proc.average_baseline_GAs()` after the three averaging methods. JS25 and MSA21 contain no pure-tone DPOAE recordings, so skip `analyze_PT_DPOAE` for them.
- **MSA21**: use `DPOAEProcessorMSA21` and `analyze_speech_DPOAE()`; for the out-of-ear recordings `analyze_speech_DPOAE_outofear()`.
- **Out-of-ear recordings** (JS25, JS26): pass `outofear=True` to `analyze_speech_DPOAE` and skip `analyze_PT_DPOAE`.

In JS26, DPOAEs were also recorded in the ear ipsilateral to the audiobook. These recordings are not evaluated with this pipeline; we are currently working on evaluation procedures for ipsilateral speech-like DPOAEs.

### 2. Sort results per condition

Set `DATA_ROOT` and check the participant lists in `DATASETS` at the top of `sort_files_to_conditions.py`, then run

```bash
python sort_files_to_conditions.py js25 js26 msa21
```

This writes `<evaluation folder>_files_per_condition.npy` to `<dataset>/Statistical_Analysis/<first>_to_<last participant>/` (for out-of-ear recordings to `<dataset>/Statistical_Analysis/`).

### 3. Statistics

`DPOAEStatistics` computes the grand averages and the values that are compared between conditions. The analyses in both papers use the following steps (`dataset` is `'JS25'`, `'JS26'` or `'MSA21'`):

```python
from DPOAE_statistics import DPOAEStatistics

stats = DPOAEStatistics(files_per_condition_path, res_path, participants)

# grand averages, with Gaussian fits of the cross-correlation envelopes
stats.compute_grand_average(method='crosscorr', fit_GA=True)
stats.compute_grand_average(method='crosscorr', per_participant=True, fit_GA=True)
stats.compute_grand_average(method='msc')
stats.compute_grand_average(method='psd')

# values per participant: evaluated at the grand-average peak, with AUC and fitted widths
cc = stats.prepare_boxplot_data_crosscorr('GA_crosscorr', ga_peak=True, fit_peaks=True, avg_participant=True)
cc_pP = stats.prepare_boxplot_data_crosscorr('GA_crosscorr', ga_peak=True, fit_peaks=True, per_participant=True, avg_participant=False)
msc = stats.prepare_boxplot_data_msc(corrected=True)
psd = stats.prepare_boxplot_data_psd()

# proportion of trials with a significant peak
stats.prepare_barplot_data_sigpeaks()
stats.prepare_barplot_data_sigpeaks(per_participant=True)

# attention modulation coefficient, AMC = 2 (attended - ignored) / (attended + ignored)
stats.convert_boxplot_data_for_amc(cc['mag'], f'AMC_data_crosscorr_mag_avgP.npy', dataset=dataset)

stats.convert_boxplot_data_for_amc(cc_pP['auc'], f'AMC_data_crosscorr_auc_pP.npy', dataset=dataset)    
    
stats.convert_boxplot_data_for_amc(msc['MSCpw'], 'AMC_data_MSCpw_corrected_avgP.npy', dataset=dataset)

stats.convert_boxplot_data_for_amc(psd['PSD_Pa'], 'AMC_data_PSD_Pa_avgP.npy', dataset=dataset)
```

All results are written to subfolders of `res_path` (`GA_*`, `Boxplot_data_*`, `Barplot_data_sigpeaks`).

Out-of-ear recordings contain no DPOAE peak, so for them the peak position is taken from the in-ear grand average. Pass the in-ear `Statistical_Analysis/<first>_to_<last>` folder as `ga_reference_path=...`; it must contain an analysis folder with the same name as `res_path`.

### 4. Plots and significance tests

`DPOAEplotter` plots the data saved in step 3:

```python
import numpy as np
from DPOAE_plotting import DPOAEplotter

plotter = DPOAEplotter(res_path / 'Plots')
cc = np.load(res_path / 'Boxplot_data_GA_crosscorr' / 'boxplot_data_crosscorr_ga_peak_fitted_avgP.npy',
             allow_pickle=True).item()
plotter.plot_boxplots_general('crosscorr', cc, related=True, sort_by='JS26_speaker')
```

- `plot_boxplots_general`: attended vs. ignored (and between voices) with paired tests: a paired t-test, or a Wilcoxon signed-rank test if the paired differences are not normally distributed (Shapiro-Wilk).
- `plot_boxplots_amc`: AMC tested against zero: a one-sample t-test, or a Wilcoxon signed-rank test if the values are not normally distributed.
- `plot_barplots_sigpeaks`: percentage of trials with a significant peak.

Box fliers (beyond 1.5 IQR) are excluded from the tests. P-values are Benjamini-Hochberg corrected across all comparisons of one metric within one call, i.e. across all figures of that metric. `sort_by` sets how the conditions are grouped into figures (see the docstring of `DPOAE_plotting.py`), e.g. `'JS26'`, `'JS26_speaker'`, `'JS25'`, `'MSA21'` or `'<dataset>_pooled_attention'`.

### Stimulus generation (JS26)

Run the scripts in `stimulus_generation/` in the order of their step numbers. Source material is a pair of audiobooks read by a male and a female speaker, from which the story and the competing speaker are taken.

1. `create_noise_segments.py` cuts segments of the other audiobook to the length of each story chapter.
2. `generate_shifted_books_and_waveforms.py` shifts the f0 of each voice to the target frequency of every harmonic condition, and derives the harmonic and DP waveforms from a band-pass-filtered fundamental waveform via the Hilbert transform (fixed bandwidth of 10 Hz for resolved and 5 Hz for unresolved harmonics).
3. `generate_multiband_stimuli.py` combines the waveforms of both voices into the stereo stimuli and mixes the audiobook chapters, with the level of the female voice matched to the male voice.

The resulting stimuli are uncalibrated; the presentation level was calibrated separately in the experimental setup.

## Citation

If you use this code, please cite the corresponding paper(s). Citation details will be added on publication.

## License

MIT, see [LICENSE](LICENSE).
