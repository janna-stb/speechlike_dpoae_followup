"""Step 2 of the JS26 stimulus generation: pitch-shift the audiobooks and create the
harmonic and DP waveforms.

The story and noise chapters of both voices are shifted to the target f0 of every harmonic
condition (`shifts`) with `shift_f0_psola`; the achieved f0 is logged to excel files. From
each shifted chapter, the harmonic waveforms (`harms`) and DP waveforms are generated with
`stimulus_generator`, using a fixed bandwidth of the fundamental waveform (`bw`, in Hz):
10 Hz for resolved and 5 Hz for unresolved harmonics. Finally, the pure-tone stimulus is
created.
"""

from datetime import datetime

from class_stimulus_gen import stimulus_generator
from functions_f0_shift import shift_f0_psola
from pathlib import Path
import glob
import librosa
import soundfile as sf
import pandas as pd
import numpy as np
     

# Root folder of the stimulus material (see README). Adjust to your local copy.
data_path = Path('path/to/stimulus_data')
results_shifted_book = data_path / 'Audiobooks/FINAL_SHIFTED'
save_stimuli_to = data_path / 'Stimuli'

books = {'Mres': 'Hund', 'Fres': 'Polarnacht', 'Munres': 'Hund', 'Funres': 'Polarnacht'}

voices = {'target': {'Hund': 'male', 'Polarnacht': 'female'},
          'noise': {'Hund': 'female', 'Polarnacht': 'male'}}

shifts = {'Mres': [140, 270],
          'Fres': [180, 325],
          'Munres': [85, 120],
          'Funres': [115, 160]}

harms = {'Mres': [[6], [8]],
         'Fres': [[6], [8]],
         'Munres': [[10, 15], [13, 18]],
         'Funres': [[10, 15], [13, 18]]}

bw = {'Mres': [10, 10],
      'Fres': [10, 10],
      'Munres': [5, 5],
      'Funres': [5, 5]}

stim_types = ['Fres', 'Mres', 'Fres', 'Munres', 'Funres']

stim_type_noise = {'Mres': 'Fres',
                   'Fres': 'Mres',
                   'Munres': 'Funres',
                   'Funres': 'Munres'}

for stim_type in stim_types:
    book = books[stim_type]
    shift_list_story = shifts[stim_type]
    shift_list_noise = shifts[stim_type_noise[stim_type]]
    harm = harms[stim_type]
    target_voice = voices['target'][book]
    noise_voice = voices['noise'][book]

    book_path_target = data_path / f'Audiobooks/FINAL_STORY_{target_voice.upper()}'
    book_path_noise = data_path / f'Audiobooks/FINAL_NOISE_{noise_voice.upper()}'
    
    # extract all .wav files
    target_files = sorted(glob.glob(str(book_path_target / '*.wav')))
    noise_files = sorted(glob.glob(str(book_path_noise / '*.wav')))

    # create shifted audiobooks first, stimuli later in a separate step
    shift_results_target = []
    for shift in shift_list_story:
        output_folder = results_shifted_book / f'STORY_{target_voice.capitalize()}' / f'f0_to_{shift}Hz'
        output_folder.mkdir(parents=True, exist_ok=True)

        for file in target_files:
            output_path = output_folder / Path(file).name

            # check if file already exists
            if output_path.exists():
                print(f'\n{Path(*output_path.parts[-3:])} already exists, skipping...')
                continue

            # load the original audio array
            audio_array, _  = librosa.load(file, sr=44100, mono=True)
            #shifted_story, shifted_f0, shifted_mean = shift_f0(audio_array, shift)
            shifted_story, shifted_f0 = shift_f0_psola(audio_array, shift)
            # reduce dimensions of shifted_story if it has dimension (1, n_samples) instead of (n_samples,) for saving
            shifted_story = np.squeeze(shifted_story) 

            print(f'Shifted f0: {shifted_f0:.2f} Hz for {Path(*output_path.parts[-3:])}\n')

            # save shifted story
            sf.write(output_path, shifted_story, 44100)
            shift_results_target.append([Path(*output_folder.parts[-2:]) / Path(file).name, shift, shifted_f0, abs(shift - shifted_f0)])
    
    if len(shift_results_target) > 0:
        # save shift results to excel
        df_shift_results_target = pd.DataFrame(shift_results_target, columns=['file', 'target_f0', 'estimated_f0', 'error'])
        excel_file_path = results_shifted_book / f'shift_results_STORY_{target_voice.upper()}_{stim_type}.xlsx'
        mode = 'a' if excel_file_path.exists() else 'w'
        # Create a timestamp-based sheet name
        sheet_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
        with pd.ExcelWriter(excel_file_path, mode=mode) as writer:
            df_shift_results_target.to_excel(writer, sheet_name=sheet_name, index=False)

    shift_results_noise = []
    for shift in shift_list_noise:
        output_folder = results_shifted_book / f'NOISE_{noise_voice.capitalize()}' / f'f0_to_{shift}Hz'
        output_folder.mkdir(parents=True, exist_ok=True)

        for file in noise_files:
            output_path = output_folder / Path(file).name

            # check if file already exists
            if output_path.exists():
                print(f'\n{Path(*output_path.parts[-3:])} already exists, skipping...')
                continue

            # load the original audio array
            audio_array, _  = librosa.load(file, sr=44100, mono=True)
            #shifted_noise, shifted_f0, shifted_mean = shift_f0_psola(audio_array, shift)
            shifted_noise, shifted_f0 = shift_f0_psola(audio_array, shift)
            shifted_noise = np.squeeze(shifted_noise)

            print(f'Shifted f0: {shifted_f0:.2f} Hz for {Path(*output_path.parts[-3:])}\n')

            # save shifted noise
            sf.write(output_path, shifted_noise, 44100)
            shift_results_noise.append([Path(*output_folder.parts[-2:]) / Path(file).name, shift, shifted_f0, abs(shift - shifted_f0)])
        
    # save shift results to excel
    if len(shift_results_noise) > 0:
        df_shift_results_noise = pd.DataFrame(shift_results_noise, columns=['file', 'target_f0', 'estimated_f0', 'error'])
        excel_file_path_noise = results_shifted_book / f'shift_results_NOISE_{noise_voice.upper()}_{stim_type}.xlsx'
        mode = 'a' if excel_file_path_noise.exists() else 'w'
        sheet_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
        with pd.ExcelWriter(excel_file_path_noise, mode=mode) as writer:
            df_shift_results_noise.to_excel(writer, sheet_name=sheet_name, index=False)
    

    for i, shift in enumerate(shift_list_story):
        bw_target = bw[stim_type][i]
        sg_target = stimulus_generator(
            n1=harm[0],
            n2=harm[1],
            source_dir=results_shifted_book / f'STORY_{target_voice.capitalize()}/f0_to_{shift}Hz',
            target_dir=save_stimuli_to / f'STORY_{target_voice.capitalize()}',
            fs=44100
        )
        sg_target.create_stimuli(bw=bw_target)

    for i, shift in enumerate(shift_list_noise):
        bw_noise = bw[stim_type][i]
        sg_noise = stimulus_generator(
            n1=harm[0],
            n2=harm[1],
            source_dir=results_shifted_book / f'NOISE_{noise_voice.capitalize()}/f0_to_{shift}Hz',
            target_dir=save_stimuli_to / f'NOISE_{noise_voice.capitalize()}',
            fs=44100
        )
        sg_noise.create_stimuli(bw=bw_noise)

    sg_target.create_pure_tone_stimuli(1000, 1200, duration=60)  # create pure tone stimuli