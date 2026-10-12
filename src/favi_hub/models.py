"""Single CLIP vision backend. Import only with the train extra installed."""

import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, encoder, method, seed, lora=None):
        super().__init__()
        self.method = method
        hidden = encoder.config.hidden_size
        encoder.requires_grad_(False)
        if method == "lora":
            from peft import LoraConfig, get_peft_model

            encoder = get_peft_model(encoder, LoraConfig(
                r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                target_modules=lora["target_modules"], bias="none"))
        self.encoder = encoder
        # Head initialization is independent of adapter initialization RNG consumption.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.head = nn.Linear(hidden, 3)
        self.check_trainable()

    def train(self, mode=True):
        super().train(mode)
        if self.method == "frozen":
            self.encoder.eval()
        return self

    def forward(self, pixels):
        features = self.encoder(pixel_values=pixels).pooler_output
        return self.head(features)

    def check_trainable(self):
        names = [n for n, p in self.named_parameters() if p.requires_grad]
        if any(not (n.startswith("head.") or (self.method == "lora" and "lora_" in n)) for n in names):
            raise ValueError("Unexpected backbone parameters are trainable")
        if self.method == "lora" and not any("lora_" in n for n in names):
            raise ValueError("LoRA attached no trainable adapter")
        return {"trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "total_params": sum(p.numel() for p in self.parameters()), "trainable_names": names}


def load_model(cfg):
    from transformers import CLIPImageProcessor, CLIPVisionModel

    p = cfg["protocol"]
    torch.manual_seed(p["seed"])
    encoder = CLIPVisionModel.from_pretrained(p["model"]["id"], revision=p["model"]["revision"])
    processor = CLIPImageProcessor.from_pretrained(p["model"]["id"], revision=p["model"]["revision"])
    if encoder.config.image_size != p["input_resolution"]:
        raise ValueError("Configured input resolution does not match pretrained encoder")
    if processor.crop_size != {"height": p["input_resolution"], "width": p["input_resolution"]}:
        raise ValueError("Processor crop does not match configured resolution")
    return Classifier(encoder, cfg["method"], p["seed"], cfg.get("lora")), processor
