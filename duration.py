import os
import wave

def get_wav_duration(file_path):
    with wave.open(file_path, 'r') as audio:
        frames = audio.getnframes()
        rate = audio.getframerate()
        duration = frames / float(rate)
    return duration

def total_duration_in_subfolders(root_folder):
    total_duration = 0.0
    for subdir, _, files in os.walk(root_folder):
        for filename in files:
            if filename.endswith('.wav'):
                file_path = os.path.join(subdir, filename)
                total_duration += get_wav_duration(file_path)
    return total_duration

root_folders = ["sagalee/train", "sagalee/dev", "sagalee/test"]
for root_folder in root_folders:
    duration_in_seconds = total_duration_in_subfolders(root_folder)
    hours = int(duration_in_seconds // 3600)
    minutes = int((duration_in_seconds % 3600) // 60)
    seconds = int(duration_in_seconds % 60)

    print(f"{root_folder} duration: {hours} hours, {minutes} minutes, {seconds} seconds")