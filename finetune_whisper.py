from transformers import WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, load_from_disk
import torch
import os
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from tqdm import tqdm
import torchaudio
import pandas as pd
from datasets import Dataset, DatasetDict
import evaluate


def load_sagalee_dataset(base_dir):
    dataset = []
    for split in ["train", "dev", "test"]:
        split_dir = os.path.join(base_dir, split)
        for speaker_id in tqdm(os.listdir(split_dir),total=len(os.listdir(split_dir)),desc=f"loading {split}"):
            speaker_dir = os.path.join(split_dir, speaker_id)
            if os.path.isdir(speaker_dir):
                for file in os.listdir(speaker_dir):
                    if file.endswith(".wav"):
                        audio_path = os.path.join(speaker_dir, file)
                        transcript_path = os.path.join(speaker_dir, file.replace('.wav', '.txt'))
                        with open(transcript_path, 'r') as f:
                            transcription = f.read().strip()
                        dataset.append({"audio": audio_path, "text": transcription, "split": split})
    return dataset

# Preprocessing: Tokenization and feature extraction
def preprocess_function(examples):
    audio_path = examples["audio"]
    speech_array, _ = torchaudio.load(audio_path)
    transcription = examples["text"]
    tokenized_input = tokenizer(transcription, return_tensors="pt").input_ids
    features = processor(speech_array.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_features
    return {
        "input_features": features.squeeze(),
        "labels": tokenized_input.squeeze(),
    }

# Load Whisper Model, Tokenizer, Processor
model_name = "whisper-large-v3"
model_path = f"openai/{model_name}" 
tokenizer = WhisperTokenizer.from_pretrained(model_path)
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)

# Freeze the bottom layers of the encoder
n_freeze_layers = 20  
for param in model.model.encoder.layers[:n_freeze_layers]:
    param.requires_grad = False

# freeze entire encoder    
# for param in model.model.encoder.parameters():
#     param.requires_grad = False

# Dataset
dataset_dir = "processed_dataset"
if os.path.exists(dataset_dir+"/train") and os.path.exists(dataset_dir+"/test") and os.path.exists(dataset_dir+"/dev"):
    print(f"Loading processed dataset from: {dataset_dir}")
    train_data = load_from_disk(dataset_dir+"/train")
    test_data = load_from_disk(dataset_dir+"/test")
    dev_data = load_from_disk(dataset_dir+"/dev")
else:
    print("Loading raw data and extracting features")
    # Prepare dataset and data loader
    base_dir = "/work103/turi/project/oasr/sagalee"  
    dataset = load_sagalee_dataset(base_dir)
    dataset = Dataset.from_pandas(pd.DataFrame(dataset))
    dataset = DatasetDict({'train': dataset.filter(lambda x: x['split'] == 'train'), 'dev': dataset.filter(lambda x: x['split']=='dev'), 'test': dataset.filter(lambda x: x['split']=='test')})
    # Preprocess data
    train_data = dataset['train'].map(preprocess_function)
    dev_data = dataset['dev'].map(preprocess_function)
    test_data = dataset['test'].map(preprocess_function)
    print(f"Preprocessed dataset\n{train_data}")
    train_data = train_data.remove_columns(['audio','text', 'split'])
    dev_data = dev_data.remove_columns(['audio','text', 'split'])
    test_data = test_data.remove_columns(['audio','text', 'split'])
    print(f"Train data columns removed: \n{train_data}")
    # Save processed data to disk for later use
    train_data.save_to_disk(dataset_dir+"/train")
    dev_data.save_to_disk(dataset_dir+"/dev")
    test_data.save_to_disk(dataset_dir+"/test")
    print(f"Saved processed dataset to {dataset_dir}")

# Custom collate function to pad sequences
def collate_fn(examples):
    input_features = [{'input_features': item['input_features']} for item in examples]
    batch = processor.feature_extractor.pad(input_features, return_tensors='pt')
    label_features = [{'input_ids': item['labels']} for item in examples]
    labels_batch = processor.tokenizer.pad(label_features, return_tensors='pt')
    # replace padding with -100 to ignore loss correctly
    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
    batch["labels"] = labels
    return batch
    

# Convert the dataset into a PyTorch DataLoader
train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=collate_fn)
dev_dataloader = DataLoader(dev_data, batch_size=2, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=2, collate_fn=collate_fn)

# Optimizer and Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
num_epochs = 20
num_training_steps = num_epochs * len(train_dataloader)//2
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=num_training_steps
)

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice being used: {device}\n")
model.to(device)
model.generation_config.task = "transcribe"

wer_metric = evaluate.load("wer")
best_val_loss = float("inf")  # Initialize with a very large value
exp_path = f"exp/{model_name}-om"
best_model_path = f"{exp_path}/best_model"

checkpoint_path = f"{exp_path}/checkpoints"
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    total_correct = 0
    total_samples = 0
    progress_bar.set_description(f"Epoch {epoch+1}")
    for batch in train_dataloader:
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)
        # Forward pass
        outputs = model(input_features=input_features, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.set_postfix(loss=loss.item())
        progress_bar.update(1)
    model.save_pretrained(f"{exp_path}/checkpoints/epoch_{epoch+1}")
    # Calculate average loss and accuracy for the epoch
    avg_loss = total_loss / len(train_dataloader)
    #avg_accuracy = total_correct / total_samples
    print(f"Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}")
    
    # Validate the model on the dev set
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in dev_dataloader:
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_features=input_features, labels=labels)
            val_loss += outputs.loss.item()
    
    avg_val_loss = val_loss / len(dev_dataloader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # Save the model if validation loss improves
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained(best_model_path)
        print(f"Best model saved with validation loss: {avg_val_loss:.4f}")

    model.train()  # Return to training mode after validation

print(f"\n\n COMPLETED TRAINING \n")

# Evaluation: Calculate Word Error Rate (WER) on the test set
model.load_pretrained(best_model_path)  # Load the best model for evaluation
model.eval()

for batch in test_dataloader:
    input_features = batch["input_features"].to(device)
    labels = batch["labels"].to(device)
    
    with torch.no_grad():
        generated_tokens = model.generate(input_features)
    
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    wer_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

final_wer = wer_metric.compute()
print(f"Word Error Rate (WER) on Test Set: {final_wer:.2f}")
