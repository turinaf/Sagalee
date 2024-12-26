# Sagalee: Automatic Speech Recognition Dataset for Oromo language

Sagalee dataset released under the CC BY-NC 4.0 International license, a summary of the license can be found [here](https://creativecommons.org/licenses/by-nc/4.0/), and the full license can be found [here](https://creativecommons.org/licenses/by-nc/4.0/legalcode).

The dataset is available [on this link](https://drive.google.com/file/d/1u3QT-s25NAnAU8XE5HaXlwfDVQPnYLtY/view?usp=sharing) 

##  News 
* 🎉 [2024-12-20] Sagalee paper accepted to [ICASS 2025](https://2025.ieeeicassp.org/) Conference


## Training ASR on Sagalee Dataset
### Clone the Repo
```
git clone https://github.com/turinaf/sagalee.git
cd sagalee
git submodule update --init --recursive
```
### Create env and install dependancy
```
conda create -n wenet python=3.10
conda activate wenet
conda install conda-forge::sox
pip install torch==2.2.2+cu121 torchaudio==2.2.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```
```
cd wenet
pip install -r requirements.txt
```
## Training recipes
 
 ### 1 Prepare the data. 
 Running the script `prepare_wenet_data.py` script will prepare data in required format inside `wenet/examples/librispeech/s0/data/`. It organize the wav files and text files into two files. `wav.scp` containing two tab-separated columns with `wav_id` and `wav_path` and `txt` containing two tab-separated columns `wav_id` and `text_label`


`wav.scp` file:
```
sagalee_SPKR232_122     sagalee/train/SPKR232/sagalee_SPKR232_122.wav
sagalee_SPKR232_002     sagalee/train/SPKR232/sagalee_SPKR232_002.wav
```
`text` file
```
sagalee_SPKR232_082     HOJJATAA JIRA JECHUUN KOMATE
sagalee_SPKR232_093     SAMMUU KEE KEESSA HIN KAAYANI
```
### 2 Run the training
After preparing data, navigate to the directory containing `run.sh` and simply run the stages starting from stage 1. 
```
cd wenet/examples/librispeech/s0
```
``` 
bash run.sh --stage 1 --stop_stage 1
bash run.sh --stage 2 --stop_stage 2
bash run.sh --stage 3 --stop_stage 3
bash run.sh --stage 4 --stop_stage 4
bash run.sh --stage 5 --stop_stage 5
```
* <strong> Stage 1</strong>: is used to extract global cmvn(cepstral mean and variance normalization) statistics. These statistics will be used to normalize the acoustic features.
* <strong> Stage 2</strong>: Generate label token dictionary
* <strong> Stage 3</strong>: This stage generates the WeNet required format file `data.list` in json format.
* <strong> Stage 4</strong>: Training 
* <strong> Stage 4</strong>: Testing the trained model
## Finetuning Whisper model
- `finetune_whisper.py` is used to fine tune whisper largev3 (you can change model size) by freezing bottom layers of encoder on Sagalee dataset, you can simply run this python script to finetune.
```
python finetune_whisper.py
```
- For full paramater finetuning, follow these [steps](https://github.com/turinaf/wenet/blob/f4ff710f95bb30bdd898fd463f2877a504df7533/examples/aishell/whisper/README.md) in wenet script.
## Acknowledgement
The training code is adapted from [WeNet](https://github.com/wenet-e2e/wenet) and used to train model on our custom [Sagalee](https://github.com/turinaf/Sagalee) Dataset.
