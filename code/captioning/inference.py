import torch
from PIL import Image

from models import build_model 
# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

CKPT = "path/to/checkpoint"
IMAGE_PATH = "path/to/image.*"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_PATCH_TOKENS = 64
MAX_NEW_TOKENS = 256 
# ---------------------------------------------------------
# Build model
# ---------------------------------------------------------


model, tokenizer, image_processor = build_model(
        image_encoder_name="hf_hub:Snarcy/tbd_b",
        decoder_type="facebook/bart-base",
        freeze_vision=True,
        lora_vision=False,
        lora_text=False,
        num_patch_tokens=NUM_PATCH_TOKENS
    )


model.to(DEVICE)
model.eval()

# ---------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------

state_dict = torch.load(f"{CKPT}/pytorch_model.bin", map_location="cpu")
missing, unexpected = model.load_state_dict(state_dict, strict=False)

print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

# ---------------------------------------------------------
# Load image
# ---------------------------------------------------------

image = Image.open(IMAGE_PATH).convert("RGB")

pixel_values = image_processor(
    images=[image],
    return_tensors="pt"
).pixel_values.to(DEVICE)

# ---------------------------------------------------------
# Generate caption
# ---------------------------------------------------------

with torch.no_grad():
    generated_ids = model.generate(
        pixel_values=pixel_values,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=5,
        length_penalty=1.0,
    )

caption = tokenizer.decode(
    generated_ids[0],
    skip_special_tokens=True
)

print("Caption:")
print(caption)
