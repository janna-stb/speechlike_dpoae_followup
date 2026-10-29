"""Step 1 of the JS26 stimulus generation: cut the competing-speaker segments.

For each chapter of a story, a segment of the same length is cut from the other audiobook
(read by the other voice), starting 30 s into it and leaving 10 s between consecutive
segments. The segments are saved as NOISE_<voice>_<chapter>.wav.
"""

from pathlib import Path
import librosa
import soundfile as sf

# Root folder of the stimulus material (see README). Adjust to your local copy.
main_path = Path('path/to/stimulus_data') / 'Audiobooks' / 'Dev'

story_path = {'Hund': main_path / 'FINAL_STORY_Hund/',
              'Polarnacht': main_path / 'FINAL_STORY_Polarnacht/'}

noise_path = {'Hund': main_path / 'CUTTED_Polarnacht_forig/merged_Polarnacht_Rest_ab_1-36-52.wav', 
              'Polarnacht': main_path / 'CUTTED_Hund_morig_V2/Hund_Rest_ab_36-03.wav'}

result_folders = {'Hund': main_path.parent / 'FINAL_NOISE_FEMALE',
                  'Polarnacht': main_path.parent / 'FINAL_NOISE_MALE'}

chapters = [str(f'0{i}') for i in range(1, 10)] + [str(i) for i in range(10, 13)]

for story in ['Polarnacht', 'Hund']:
    start = int(44100 * 60 * 0.5) # start after 0.5 minutes so the story has progressed a little
    noise_seg_list = []
    noise_audio, sr_noise = librosa.load(noise_path[story], sr=None)
    if story == 'Polarnacht':
        voice = 'M'
    elif story == 'Hund':
        voice = 'F'
    else:
        raise ValueError(f"Unexpected story name: {story}. Expected 'Polarnacht' or 'Hund'.")
    
    if sr_noise != 44100:
        raise ValueError(f"Sampling rate of noise audio ({sr_noise} Hz) does not match the expected 44100 Hz.")
    
    for ch in chapters:
        story_audio, sr_story = librosa.load(story_path[story] / f'{story}_{ch}.wav', sr=None)
        if sr_story != 44100:
            raise ValueError(f"Sampling rate of story audio ({sr_story} Hz) does not match the expected 44100 Hz.")
        
        # determine length of segment
        segment_length = len(story_audio)
        end = start + segment_length

        # extract noise segment
        noise_segment = noise_audio[start:end]
        noise_seg_list.append(noise_segment)

        # update start for next segment
        start = end + 44100 * 10  

    for i, ch in enumerate(chapters):
        # save noise segment
        noise_segment = noise_seg_list[i]
        # ensure result folder exists
        result_folder = result_folders[story]
        result_folder.mkdir(parents=True, exist_ok=True)

        sf.write(result_folder / f'NOISE_{voice}_{ch}.wav', noise_segment, sr_story)
