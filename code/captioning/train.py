import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

from models import build_model   
# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DOWNLOAD_PATH_DATASET = "path/to/dataset/cache"
OUT_PATH = "path/to/output/checkpoints"
NUM_PATCH_TOKENS = 64
os.makedirs(DOWNLOAD_PATH_DATASET, exist_ok=True)
os.makedirs(OUT_PATH, exist_ok=True)

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

ds = load_dataset(
    "eltorio/ROCOv2-radiology",
    split={
        "train": "train",
        "validation": "validation",
        "test": "test"
    },
    cache_dir=DOWNLOAD_PATH_DATASET
)

# ---------------------------------------------------------
# Build model
# ---------------------------------------------------------


model, tokenizer, image_processor = build_model(
        #image_encoder_name="hf_hub:Snarcy/tbd_b",
        #image_encoder_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        #image_encoder_name="facebook/dinov2-base",
        image_encoder_name="openai/clip-vit-base-patch16",
        #decoder_type="facebook/bart-base",
        decoder_type="facebook/bart-large",
        freeze_vision=True,
        lora_vision=False,
        lora_text=False,
        num_patch_tokens=NUM_PATCH_TOKENS,
    )


# ---------------------------------------------------------
# Dataset transform
# ---------------------------------------------------------

def transforms(batch):
    images = [img.convert("RGB") for img in batch["image"]]
    captions = batch["caption"]

    pixel_values = image_processor(
        images=images,
        return_tensors="pt"
    ).pixel_values

    tokenized = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    labels = tokenized.input_ids.clone()
    labels[tokenized.attention_mask == 0] = -100

    return {
        "pixel_values": pixel_values,
        "input_ids": tokenized.input_ids,
        "attention_mask": tokenized.attention_mask,
        "labels": labels,
    }

# ---------------------------------------------------------
# Apply transforms
# ---------------------------------------------------------

train_ds = ds["train"].shuffle(seed=42)
val_ds = ds["validation"]
test_ds = ds["test"]

train_ds.set_transform(transforms)
val_ds.set_transform(transforms)
test_ds.set_transform(transforms)

# ---------------------------------------------------------
# Training arguments
# ---------------------------------------------------------

training_args = TrainingArguments(
    output_dir=OUT_PATH,
    learning_rate=5e-5,
    num_train_epochs=20,
    fp16=True,
    warmup_steps=2500,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=4,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    logging_steps=25,
    remove_unused_columns=False,
    push_to_hub=False,
    save_safetensors=False,
    label_names=["labels"],
    load_best_model_at_end=True,
    report_to="none"
)

# ---------------------------------------------------------
# Trainer
# ---------------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

# ---------------------------------------------------------
# Train
# ---------------------------------------------------------

trainer.train(resume_from_checkpoint=False)
