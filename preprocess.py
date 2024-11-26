import os
import re
from tqdm import tqdm

root_dir = "sagalee"
def remove_punc(sentence):
    # remove some special characters
    sentence = sentence.replace("è", "e").replace("ₒ", "").replace("•", "").replace("ʼ", "").replace("''", "").replace("_", " ").replace('\xa0', " ")
    # Remove punc while retaining apostrophe and dot in decimal numbers
    sentence = re.sub(r"(?!\b'\b)(?<!\d)\.(?!\d)|[^\w\s'.]", "", sentence).replace(" '", " ").replace("' ", " ").strip("'")
    # Replace two or more spaces with single space
    sentence = re.sub(r'\s{2,}', " ", sentence)
    return sentence.strip()

for root, _, files in os.walk(root_dir):
    for filename in tqdm(files, total=len(files), desc=f"Root: {root}"): 
        if filename.endswith(".txt"):
            txt_path = os.path.join(root, filename)
            with open(txt_path, 'r', encoding='utf-8') as f:
                sentence = f.read().strip()
            sentence = remove_punc(sentence)
            with open(txt_path, 'w', encoding='utf-8') as outf: 
                outf.write(sentence)

print("DONE!!")

