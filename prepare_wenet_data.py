import os
from tqdm import tqdm
source_dir = "sagalee"
output_dir = "wenet/examples/aishell/s0/data"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for folder in os.listdir(source_dir):
    target_path = os.path.join(output_dir, folder) 
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    subdir = os.path.join(source_dir, folder)
    if not os.path.isdir(subdir):
       continue
    for root, _, files in os.walk(subdir):
        for filename in tqdm(files, total=len(files), desc=f"Processing{root}: "):
            if filename.endswith(".wav"):
                wav_id = filename.split(".")[0]
                wav_path = os.path.join(root, filename)
                wav_abs_path = os.path.abspath(wav_path)
                text_path = os.path.join(root, wav_id+".txt")
                with open(text_path, "r", encoding='utf-8') as rf:
                    label = rf.read()
                with open(f"{target_path}/text", "a", encoding="utf-8") as wf:
                    wf.write(wav_id+"\t"+label+"\n")
                with open(f"{target_path}/wav.scp", "a", encoding="utf-8") as sf:
                    sf.write(wav_id+"\t"+wav_abs_path+"\n")