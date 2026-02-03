<div align="center">

# 🏥 OmniRad

**A Unified Framework for Medical Image Analysis**

*Classification • Dense Prediction (Segmentation) • Captioning*

</div>


## 🔍 Overview

**OmniRad** is a comprehensive framework for medical image analysis that supports three main tasks:

| Task | Description | Backbone Support |
|------|-------------|------------------|
| **Classification** | Multi-class medical image classification with LoRA support | ViT, DINOv2, DINOv3, CLIP, RadioDino |
| **Dense Prediction** | Medical image segmentation using HybridEncoder (DEIMv2-based) | ViT, DINOv2, DINOv3, RadioDino |
| **Captioning** | Radiology report generation with vision-language models | ViT + BART |

> 🤗 **Pre-trained models are available on HuggingFace**: Check out our [model collection](https://huggingface.co/collections/Snarcy/omnirad) for pre-trained vision backbones.

---

## 🏗️ Architecture

<div align="center">
<img src="res/architecture.png" alt="OmniRad Architecture" width="60%">
</div>

The OmniRad architecture leverages:
- **Backbone**: Pre-trained Vision Transformers (ViT) with optional freezing
- **STA (Spatial Token Aggregation)**: Multi-scale feature extraction from ViT layers
- **HybridEncoder**: DEIMv2-style encoder for dense prediction tasks
- **Task-specific Heads**: Classification head, Segmentation decoder, or BART decoder for captioning

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/unica-visual-intelligence-lab/OmniRad.git
cd OmniRad

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

**Requirements include:**
- Core ML frameworks: PyTorch, torchvision, transformers
- Vision models: timm, accelerate
- Medical imaging: medsegbench, segmentation-models-pytorch
- Utilities: PEFT (LoRA), datasets, scikit-learn, tqdm, PyYAML
- Training tools: tensorboard, jiwer, evaluate

---

## 🚀 Training

### 1. Classification

The classification module supports fine-tuning pre-trained models on medical imaging datasets with optional LoRA adapters.

#### Single Model Training

```bash
cd code/classification

python train.py \
    --model "hf_hub:Snarcy/RadioDino-s16" \
    --dataset_path_train /path/to/train \
    --dataset_path_val /path/to/val \
    --dataset_path_test /path/to/test \
    --output_path ./outputs \
```

#### With LoRA Adapters (Parameter-Efficient Fine-Tuning)

```bash
python train.py \
    --model "hf_hub:Snarcy/RadioDino-s16" \
    --dataset_path_train /path/to/train \
    --dataset_path_val /path/to/val \
    --dataset_path_test /path/to/test \
    --output_path ./outputs \
    --dataset_name "breastmnist" \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1
```

#### Classification Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `hf_hub:Snarcy/RadioDino-s16` | Model name (timm-compatible) |
| `--epochs` | 40 | Number of training epochs |
| `--batch_size` | 128 | Batch size |
| `--lr` | 1e-5 | Learning rate |
| `--warmup_epochs` | 10 | Warmup epochs |
| `--dropout` | 0.4 | Dropout rate |
| `--use_lora` | False | Enable LoRA adapters |
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha |
| `--use_amp` | True | Automatic mixed precision |
| `--gradient_clip` | 1.0 | Gradient clipping value |
| `--patience` | 10000 | Early stopping patience |
| `--cutmix` | False | Use CutMix augmentation |
| `--mixup` | False | Use MixUp augmentation |

#### Batch Training (Multiple Models/Datasets)

Edit `launch_finetuning.py` to configure your experiments:

```python
MODELS = [
    "hf_hub:Snarcy/tbd_s",
    "hf_hub:Snarcy/tbd_b",
]

DATASETS = [
    "breastmnist",
    "pneumoniamnist",
    "organamnist",
]
```

Then run:
```bash
python launch_finetuning.py
```

### 2. Dense Prediction (Segmentation)

The dense prediction module is based on **DEIMv2** architecture, using a HybridEncoder with STA (Spatial Token Aggregation) for medical image segmentation.

#### Training with Base Models

```bash
cd code/dense

python tools/train_segmentation.py -c ./configs/segmentation/medseg_base_models.yml
```

#### Training with Small Models

```bash
python tools/train_segmentation.py -c ./configs/segmentation/medseg_small_models.yml
```

#### Custom Configuration

You can override config values via command line:

```bash
python tools/train_segmentation.py \
    -c ./configs/segmentation/medseg_base_models.yml \
    --output-dir ./custom_outputs \
    --use-amp
```

#### Segmentation Config Example (`medseg_base_models.yml`)

```yaml
root: 'path_to_dense_root'

# Models to benchmark
models:
  - 'hf_hub:Snarcy/tbd_b'
  - 'hf_hub:Snarcy/RadioDino-b16'
  - 'vit_base_patch16_dinov3_qkvb.lvd1689m'
  - 'vit_base_patch14_dinov2'
  - 'vit_base_patch16_224'

freeze_backbone: true  # Freeze ViT backbone, train only STA + HybridEncoder + decoder
image_size: [448, 448]

# Training hyperparameters
epochs: 20
batch_size: 16
workers: 2
lr: 0.0001
use_amp: true

output_dir: './outputs/medseg_benchmark'

# STA config
use_sta: true
hidden_dim: 768  # ViT embed_dim
out_indices: [3, 7, 11]

# HybridEncoder config (DEIMv2-style)
use_hybrid_encoder: true
HybridEncoder:
  in_channels: [768, 768, 768]
  hidden_dim: 768
  nhead: 6
  enc_act: 'gelu'
  version: 'deim'

# Datasets
datasets:
  - BusiMSBench
  - CovidQUExMSBench
  - Promise12MSBench
```

#### Segmentation Config Variants

| Config | Hidden Dim | Model Size | Use Case |
|--------|-----------|------------|----------|
| `medseg_base_models.yml` | 768 | Base (~86M) | Best performance |
| `medseg_small_models.yml` | 384 | Small (~22M) | Resource-constrained |

---

### 3. Captioning

The captioning module generates radiology reports using a vision-language architecture with BART decoder.

#### Training

Edit paths in `code/captioning/train.py`:

```python
DOWNLOAD_PATH_DATASET = "path/to/dataset/cache"
OUT_PATH = "path/to/output/checkpoints"
```

Then run:

```bash
cd code/captioning
python train.py
```

#### Configuration in `train.py`

```python
model, tokenizer, image_processor = build_model(
    image_encoder_name="hf_hub:Snarcy/tbd_b",  # Vision encoder
    decoder_type="facebook/bart-large",         # Text decoder
    freeze_vision=True,                         # Freeze vision encoder
    lora_vision=False,                          # LoRA for vision
    lora_text=False,                            # LoRA for text
    num_patch_tokens=64,                        # Patch token reduction
)
```

#### Training Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 5e-5 | Learning rate |
| `num_train_epochs` | 20 | Training epochs |
| `per_device_train_batch_size` | 16 | Training batch size |
| `gradient_accumulation_steps` | 4 | Gradient accumulation |
| `warmup_steps` | 2500 | Warmup steps |
| `fp16` | True | Mixed precision |
| `eval_steps` | 500 | Evaluation frequency |

#### Supported Vision Encoders

- `hf_hub:Snarcy/tbd_b` - RadioDino Base
- `facebook/dinov2-base` - DINOv2 Base
- `openai/clip-vit-base-patch16` - CLIP ViT-B/16

#### Supported Text Decoders

- `facebook/bart-base` - BART Base
- `facebook/bart-large` - BART Large

---

## 🔮 Inference

### Classification Inference

```python
import torch
import timm
from PIL import Image
from torchvision.transforms import v2

# Load model
model = timm.create_model("hf_hub:Snarcy/RadioDino-s16", pretrained=True, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("path/to/best.pth"))
model.eval()

# Transform
transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Predict
image = Image.open("image.png").convert("RGB")
input_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    output = model(input_tensor)
    pred = torch.argmax(output, dim=1)
```

### Captioning Inference

```python
import torch
from PIL import Image
from models import build_model

# Build and load model
model, tokenizer, image_processor = build_model(
    image_encoder_name="hf_hub:Snarcy/tbd_b",
    decoder_type="facebook/bart-base",
    freeze_vision=True,
    num_patch_tokens=64
)

state_dict = torch.load("path/to/checkpoint/pytorch_model.bin", map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()

# Generate caption
image = Image.open("xray.png").convert("RGB")
pixel_values = image_processor(images=[image], return_tensors="pt").pixel_values

with torch.no_grad():
    generated_ids = model.generate(
        pixel_values=pixel_values,
        max_new_tokens=256,
        num_beams=5,
        length_penalty=1.0,
    )

caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(caption)
```

---

## 🤖 Supported Models

### Vision Backbones

| Model | Size | Patch Size | Source |
|-------|------|------------|--------|
| `hf_hub:Snarcy/tbd_s` | Small | 16 | HuggingFace |
| `hf_hub:Snarcy/tbd_b` | Base | 16 | HuggingFace |
| `hf_hub:Snarcy/RadioDino-s16` | Small | 16 | HuggingFace |
| `hf_hub:Snarcy/RadioDino-b16` | Base | 16 | HuggingFace |
| `vit_small_patch16_dinov3_qkvb.lvd1689m` | Small | 16 | timm |
| `vit_base_patch16_dinov3_qkvb.lvd1689m` | Base | 16 | timm |
| `vit_small_patch14_dinov2` | Small | 14 | timm |
| `vit_base_patch14_dinov2` | Base | 14 | timm |
| `vit_base_patch16_clip_224.laion2b` | Base | 16 | timm |
| `vit_base_patch16_224.mae` | Base | 16 | timm |
| `vit_base_patch16_224.dino` | Base | 16 | timm |

---

## 🙏 Acknowledgements

This project builds upon several excellent works:

- **[DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)** - Real-Time Object Detection framework used as base for dense prediction
- **[timm](https://github.com/huggingface/pytorch-image-models)** - PyTorch Image Models library
- **[HuggingFace Transformers](https://github.com/huggingface/transformers)** - BART models for captioning
- **[PEFT](https://github.com/huggingface/peft)** - Parameter-Efficient Fine-Tuning (LoRA)
- **[RadioDino](https://huggingface.co/Snarcy)** - Pre-trained medical imaging models

---

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ by [UNICA Visual Intelligence Lab](https://github.com/unica-visual-intelligence-lab)**

</div>