import torch
import torch.nn as nn
import timm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoImageProcessor,
    AutoModel,

)

# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def print_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total}")
    print(f"Trainable params: {trainable}")

# ---------------------------------------------------------
# Vision encoder
# ---------------------------------------------------------

class VisionEncoder(nn.Module):
    def __init__(self, model_name, out_dim, freeze=True):
        super().__init__()
        # Load timm model
        if model_name.startswith("hf_hub:"):
            self.model = timm.create_model(model_name, pretrained=True)
        else:
            self.model = AutoModel.from_pretrained(model_name)
        if hasattr(self.model, "num_features"):
            feat_dim = self.model.num_features
        elif hasattr(self.model.config, "hidden_size"):
            feat_dim = self.model.config.hidden_size
        else:
            feat_dim = 768  # default fallback
        self.proj = nn.Linear(feat_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        nn.init.xavier_uniform_(self.proj.weight)  # smaller init

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, pixel_values):
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(pixel_values)
        elif hasattr(self.model, "vision_model"):
            feats = self.model.vision_model(pixel_values).last_hidden_state
        else:
            feats = self.model(pixel_values).last_hidden_state
        feats = self.proj(feats)
        feats = self.norm(feats)
        return feats

# ---------------------------------------------------------
# Patch merger
# ---------------------------------------------------------

class PatchMerger(nn.Module):
    def __init__(self, dim, num_tokens_out):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(num_tokens_out, dim))

    def forward(self, x):
        x = self.norm(x)
        attn = torch.matmul(self.queries, x.transpose(-1,-2))
        attn = attn.softmax(dim=-1)
        return torch.matmul(attn, x)

# ---------------------------------------------------------
# Image-to-Text with BART
# ---------------------------------------------------------
def lora_compontents(lora_vision=False, lora_text=False, model_vision=None, model_text=None):
    if lora_vision or lora_text:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError("Please install the 'peft' library to use LoRA.")
        if lora_vision:
            vision_lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules='all-linear',
                lora_dropout=0.1,
            )
            model_vision = get_peft_model(
                model_vision,
                vision_lora_config
            )
        if lora_text:
            text_lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules='all-linear',
                lora_dropout=0.1,
            )
            model_text = get_peft_model(
                model_text,
                text_lora_config
            )
        print("LoRA components added to the model.")
    return model_vision, model_text
class ImagePrefixBART(nn.Module):
    def __init__(
        self,
        vision_encoder_name,
        decoder_name="facebook/bart-base",
        freeze_vision=True,
        num_patch_tokens=32,
        lora_vision=False,
        lora_text=False,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        #get decoder config

        self.decoder = AutoModelForSeq2SeqLM.from_pretrained(decoder_name)
        hidden = self.decoder.config.d_model
        self.num_patch_tokens = num_patch_tokens
        self.vision_encoder = VisionEncoder(
            vision_encoder_name,
            out_dim=hidden,
            freeze=freeze_vision
        )
        if num_patch_tokens is not None:
            self.patch_merger = PatchMerger(hidden, num_tokens_out=num_patch_tokens)
            print(f"PatchMerger added: reducing to {num_patch_tokens} tokens.")
        else:
            self.patch_merger = None
            print("No PatchMerger added; using all image tokens.")

        self.image_processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov3-vits16-pretrain-lvd1689m"
        )

        # Configure decoder start, pad, eos tokens
        self.decoder.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.decoder.config.pad_token_id = self.tokenizer.pad_token_id
        self.decoder.config.eos_token_id = self.tokenizer.eos_token_id

        # Add LoRA if requested
        self.vision_encoder, self.decoder = lora_compontents(
            lora_vision=lora_vision,
            lora_text=lora_text,
            model_vision=self.vision_encoder,
            model_text=self.decoder
        )
        print_params(self)
    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        labels=None,
        num_items_in_batch=None,  # Accept and ignore for Trainer
        **kwargs
    ):
        assert pixel_values is not None, "pixel_values must be provided"

        # Encode image
        img_tokens = self.vision_encoder(pixel_values)
        if self.patch_merger is not None:
            img_tokens = self.patch_merger(img_tokens)




        # Pass through decoder's encoder
        encoder_outputs = self.decoder.get_encoder()(
            inputs_embeds=img_tokens,
            attention_mask=torch.ones(img_tokens.size()[:2], device=img_tokens.device)
        )
        # Compute decoder outputs and loss
        outputs = self.decoder(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            **kwargs
        )
        return outputs  # includes .loss if labels are provided

    @torch.no_grad()
    def generate(self, pixel_values, max_new_tokens=64, num_beams=5, length_penalty=1.0):
        img_tokens = self.vision_encoder(pixel_values)
        if self.patch_merger is not None:
            img_tokens = self.patch_merger(img_tokens)


        B = img_tokens.size(0)
        encoder_attention_mask = torch.ones(
            img_tokens.size()[:2],
            device=img_tokens.device,
            dtype=torch.long
        )

        encoder_outputs = self.decoder.get_encoder()(
            inputs_embeds=img_tokens,
            attention_mask=encoder_attention_mask
        )

        decoder_input_ids = torch.full(
            (B, 1),
            self.tokenizer.bos_token_id,
            dtype=torch.long,
            device=img_tokens.device
        )

        return self.decoder.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )


# ---------------------------------------------------------
# Builder
# ---------------------------------------------------------

def build_model(
    image_encoder_name,
    decoder_type="facebook/bart-base",
    num_patch_tokens=None,
    freeze_vision=True,
    lora_vision=False,
    lora_text=False
):
    model = ImagePrefixBART(
        vision_encoder_name=image_encoder_name,
        decoder_name=decoder_type,
        num_patch_tokens=num_patch_tokens,
        freeze_vision=freeze_vision,
        lora_vision=lora_vision,
        lora_text=lora_text
    )
    return model, model.tokenizer, model.image_processor

# ---------------------------------------------------------
# Test run
# ---------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ImagePrefixBART(
        "hf_hub:Snarcy/tbd_b",
        decoder_name="facebook/bart-base",
        freeze_vision=True,
        lora_vision=False,
        lora_text=False,
        num_patch_tokens=None,
    ).to(device)
    print_params(model)

    dummy_img = torch.randn(1, 3, 224, 224).to(device)
    out = model.generate(dummy_img, max_new_tokens=30)
    print("Generated token IDs:", out)
    text = model.tokenizer.decode(out[0], skip_special_tokens=True)
    print("Generated text:", text)
