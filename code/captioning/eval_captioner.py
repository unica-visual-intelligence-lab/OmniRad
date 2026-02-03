import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import evaluate

from models import build_model  


bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")
MAX_NEW_TOKENS = 64
NUM_PATCH_TOKENS = 64

# ---------------------------------------------------------
# Metrics on dataset already transformed with "transforms"
# ---------------------------------------------------------
@torch.no_grad()
def compute_metrics_on_test_batch(model, tokenizer, test_ds, batch_size=16, device="cuda"):
    preds = []
    refs = []

    batch_pixel_values = []
    batch_refs = []

    pad_id = tokenizer.pad_token_id

    for item in tqdm(test_ds, desc="Evaluating"):
        batch_pixel_values.append(item["pixel_values"])
        batch_refs.append(item["labels"])

        if len(batch_pixel_values) == batch_size:
            pixel_values = torch.stack(batch_pixel_values).to(device)

            gen_ids = model.generate(
                pixel_values=pixel_values,
                max_new_tokens=MAX_NEW_TOKENS
            )

            gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            refs_ids = torch.stack(batch_refs).clone()
            refs_ids[refs_ids == -100] = pad_id
            true_texts = tokenizer.batch_decode(refs_ids, skip_special_tokens=True)

            preds.extend(gen_texts)
            refs.extend(true_texts)

            batch_pixel_values.clear()
            batch_refs.clear()

    if len(batch_pixel_values) > 0:
        pixel_values = torch.stack(batch_pixel_values).to(device)

        gen_ids = model.generate(
            pixel_values=pixel_values,
            max_new_tokens=MAX_NEW_TOKENS
        )

        gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        refs_ids = torch.stack(batch_refs).clone()
        refs_ids[refs_ids == -100] = pad_id
        true_texts = tokenizer.batch_decode(refs_ids, skip_special_tokens=True)

        preds.extend(gen_texts)
        refs.extend(true_texts)

    bleu = bleu_metric.compute(predictions=preds, references=[[r] for r in refs])
    rouge = rouge_metric.compute(predictions=preds, references=refs)
    meteor = meteor_metric.compute(predictions=preds, references=refs)

    return {
        "bleu": bleu,
        "rouge": rouge,
        "meteor": meteor
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    # Define paths as variables for anonymization
    CHECKPOINT = "<CHECKPOINT_PATH>"
    OUTPUT_PATH = "<OUTPUT_PATH>"
    DATA_DIR = "<DATA_DIR>"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset...")
    ds = load_dataset(
        "eltorio/ROCOv2-radiology",
        split={"test": "test"},
        cache_dir=DATA_DIR
    )
    test_ds = ds["test"]

    print("Building model...")

    model, tokenizer, image_processor = build_model(
        image_encoder_name="hf_hub:Snarcy/tbd_b",
        #image_encoder_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        #image_encoder_name="facebook/dinov2-base",
        #image_encoder_name="openai/clip-vit-base-patch16",
        #decoder_type="facebook/bart-base",
        decoder_type="facebook/bart-large",
        freeze_vision=True,
        lora_vision=False,
        lora_text=True,
        num_patch_tokens=NUM_PATCH_TOKENS,
    )
    
    

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
            "labels": labels
        }

    test_ds.set_transform(transforms)

    print("Loading checkpoint...")
    state_path = os.path.join(CHECKPOINT, "pytorch_model.bin")
    state_dict = torch.load(state_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.to(DEVICE)
    model.eval()

    print("Starting evaluation...")
    results = compute_metrics_on_test_batch(
        model=model,
        tokenizer=tokenizer,
        test_ds=test_ds,
        batch_size=32,
        device=DEVICE
    )

    print("\nFinal results:")
    for k, v in results.items():
        print(k, ":", v)

    try:
        with open(OUTPUT_PATH, "a") as f:
            f.write("\nResults for checkpoint: " + CHECKPOINT + "\n")
            for k, v in results.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")
        print("Results saved to", OUTPUT_PATH)
    except Exception as e:
        print("Failed to save results:", e)
