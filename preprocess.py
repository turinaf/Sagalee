import os
import re
from tqdm import tqdm

root_dir = "sagalee"
def remove_punc(sentence):
    # This pattern keeps integers, floating-point numbers, and words intact while removing other punctuation
    pattern = r'(?<!\d)[.,;:!?\'‘’“”](?!\d)|(?<=\D)[.,;:!?\'‘’“”](?=\d)|[^\w\s.\d]'
    # Replace all occurrences of the pattern with an empty string
    return re.sub(pattern, '', sentence)

for root, _, files in os.walk(root_dir):
    for filename in tqdm(files, total=len(files), desc=f"{root}: "):
        if filename.endswith(".txt"):
            txt_path = os.path.join(root, filename)
            with open(txt_path, "r", encoding="utf-8") as infile:
                content = infile.read()
            clean_txt = remove_punc(content).strip()
            upper = clean_txt.upper().replace("ₒ", "").replace("_", "").replace("È", "E").replace("ʼ", "").replace(" ", " ").strip()
            with open(txt_path, "w", encoding="utf-8") as outfile:
                outfile.write(upper)

print("Done")

