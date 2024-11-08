import os
import time
import pandas as pd
from datasets import Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, TrainingArguments, Trainer
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import sys



start_time = time.time()

version = 18
MODEL_PATH = "D:/whisper_modelv17"
# Ensure log file path exists and print an error if it doesn't
LOG_PATH = "D:/log"
DATASET_PATH = "D:/processed_dataset_combined.csv"
MODEL_OUTPUT_DIR = "D:/whisper_modelv18"

#Set process title
os.system(f"title AppWorks - Whisper Training - {version}")


if not os.path.exists(MODEL_PATH) or not os.path.exists(MODEL_OUTPUT_DIR) or not os.path.exists(LOG_PATH) or not os.path.exists(DATASET_PATH):
    print("model path or model output path invalid")
    sys.exit(1)

try:
    log_file = open(LOG_PATH, "w")
except Exception as e:
    raise RuntimeError(f"Failed to set up log file: {e}")

print(f"Starting training script for version: {version}...")

# Load your dataset
print("Loading dataset from CSV file.")
df = pd.read_csv(DATASET_PATH)
df['filename'] = 'content/audio/' + df['filename'].astype(str)
dataset = Dataset.from_pandas(df)
print(f"Dataset loaded with {len(df)} records.")

# Load the model
if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}.")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
    processor = WhisperProcessor.from_pretrained(MODEL_PATH)
else:
    print(f"COULD NOT LOAD MODEL FROM THIS PATH {MODEL_PATH}" )
    sys.exit(1)

print("Model and processor loaded successfully.")

# Build Training Arguments
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_dir="/content/logs",
    learning_rate=5e-5,
    num_train_epochs=1,
    remove_unused_columns=False,
    fp16=True,
)
print("Training arguments set up.")

# Data Collator
def data_collator(batch):
    print("Starting data collation for a batch.")
    audio_features = []
    labels = []

    for item in batch:
        try:
            # Check if the file exists
            if not os.path.exists(item["filename"]):
                print(f"File does not exist: {item['filename']}")
                continue

            # Try to load the audio file
            waveform, sample_rate = torchaudio.load(item["filename"])

            # Ensure the waveform is mono and resample to 16kHz if necessary
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate != 16000:
                waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

            # Process the audio sample and generate input features for the model
            input_features = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.squeeze()
            if input_features.dim() == 1:
                input_features = input_features.unsqueeze(0)
            input_features = torch.nn.functional.pad(input_features, (0, max(0, 3000 - input_features.size(-1)), 0, max(0, 80 - input_features.size(-2))))
            input_features = input_features[:80, :3000]

            audio_features.append(input_features)

            # Process the transcription text
            label_ids = processor(text=item["transcription"], return_tensors="pt", padding=True).input_ids
            labels.append(label_ids.squeeze())

        except Exception as e:
            print(f"Error processing {item['filename']}: {e}")
            continue

    if audio_features:
        input_features = torch.stack(audio_features, dim=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        print(f"Batch collated with input_features shape: {input_features.shape}")
    else:
        print("No valid audio files in this batch.")
        input_features = torch.empty(0)  # Create an empty tensor for input features if none are valid
        labels = torch.empty(0)  # Same for labels

    return {"input_features": input_features, "labels": labels}

# Set up trainer
print("Setting up the trainer.")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# Start training
print("Starting model training...")
trainer.train()

# Save the model
print("Saving fine-tuned model.")
model.save_pretrained(MODEL_OUTPUT_DIR)
processor.save_pretrained(MODEL_OUTPUT_DIR)

#Calc how long the process took
end_time = time.time()
elapsed_time = end_time - start_time

# Convert elapsed time to hours, minutes, and seconds
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
# Process Finished Message
print("=============================================")
print(f"Model - {version} saved successfully to {MODEL_OUTPUT_DIR}.")
print(f"Process completed in {int(hours):02}:{int(minutes):02}:{int(seconds):02} (hh:mm:ss).")
print("=============================================")

