"""Step 3 of the JS26 stimulus generation: create the stereo stimuli and the mixed audiobooks.

For each harmonic condition (`labels`) and story, the harmonic waveforms of the story voice
and of the competing voice are combined into the stereo stimuli with
`multiband_generator.create_multiband_competingspeaker`, and the corresponding audiobook
chapters are mixed with `mix_books` (competing voice starting after `offset` s, female voice
level-matched to the male voice with the level differences in `dBdiff_shift`).

Author: Janna Steinebach
"""

from pathlib import Path
from class_stimulus_gen import multiband_generator

# Root folder of the stimulus material (see README). Adjust to your local copy.
data_path = Path('path/to/stimulus_data')
audiobook_path = data_path / 'Audiobooks/FINAL_SHIFTED'
save_stimuli_to = data_path / 'Stimuli'

combinations = {'Res': {'Hund': [[140, 180], [270, 325]],
                        'Polarnacht': [[180, 140], [325, 270]]},
                'Unres': {'Hund': [[85, 115], [120, 160]],
                          'Polarnacht': [[115, 85], [160, 120]]}}

labels = {'Res': ['low_res', 'trans_res'],
          'Unres': ['low_unres', 'trans_unres']}

voices = {'target': {'Hund': 'male', 'Polarnacht': 'female'},
          'noise': {'Hund': 'female', 'Polarnacht': 'male'}}

harms = {'Res': [[6, 8], [6, 8]],
         'Unres': [[10, 13], [15, 18]]}

dBdiff_shift = {'STORY_FEMALE':{'115': 2.0,
                                '160': 1.7,
                                '180': 1.6,
                                '325': 1.4},
                'NOISE_FEMALE':{'115': 3.1,
                                '160': 2.0,
                                '180': 2.0,
                                '325': 1.8},
                'STORY_MALE':{'85': 0,
                              '120': 0,
                              '140': 0,
                              '270': 0},
                'NOISE_MALE':{'85': 0,
                              '120': 0,
                              '140': 0,
                              '270': 0}}

harmonics = ['Res', 'Unres']
books = ['Polarnacht', 'Hund']

nrm = 0.4

for book in books:
    target_voice = voices['target'][book]
    noise_voice = voices['noise'][book]
    # Book A is always target story
    for harm_type in harmonics:
        for i, combo in enumerate(combinations[harm_type][book]):
            harm = harms[harm_type][i]
            save_label = labels[harm_type][i]
            scaling_bookA = dBdiff_shift[f'STORY_{target_voice.upper()}'][str(combo[0])]
            scaling_bookB = dBdiff_shift[f'NOISE_{noise_voice.upper()}'][str(combo[1])]
            ratio_bookA = 10 ** (scaling_bookA / 20)
            ratio_bookB = 10 ** (scaling_bookB / 20)
            nrm_bookA = nrm / ratio_bookA
            nrm_bookB = nrm / ratio_bookB
        
            mg = multiband_generator(
                stim_directory = save_stimuli_to,
                bookA = f'STORY_{target_voice.capitalize()}/f0_to_{combo[0]}Hz',
                bookB = f'NOISE_{noise_voice.capitalize()}/f0_to_{combo[1]}Hz',
                fs=44100
            )
            target_dir_stimuli = save_stimuli_to.parent / 'Uncalibrated_Stimuli' / f'{save_label}' / f"competing_stimuli_{mg.bookA.split('/')[0]}_{mg.bookB.split('/')[0]}/{mg.bookA.split('/')[0].split('_')[0]}_{mg.bookA.split('/')[1].split('_')[2]}_{mg.bookB.split('/')[0].split('_')[0]}_{mg.bookB.split('/')[1].split('_')[2]}"
            target_dir_mixed_books = save_stimuli_to.parent / f"Mixed_Audiobooks_{mg.bookA.split('/')[0]}_{mg.bookB.split('/')[0]}/{save_label}"
            mg.create_multiband_competingspeaker(mg.bookA, mg.bookB, harm[0], harm[1], harm[0], harm[1], target_dir=target_dir_stimuli, nrm=nrm)
            mg.mix_books(audiobook_path, bookA=mg.bookA, bookB=mg.bookB, nrm=[nrm_bookA, nrm_bookB, nrm], offset=5, target_dir=target_dir_mixed_books)


