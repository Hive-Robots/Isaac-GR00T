from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoProcessor


EAGLE_LOCAL_PATH = (
    Path(__file__).resolve().parents[2]
    / "gr00t"
    / "model"
    / "modules"
    / "nvidia"
    / "Eagle-Block2A-2B-v2"
)



def _resolve_eagle_path(model_path: str) -> str:
    if model_path == "nvidia/Eagle-Block2A-2B-v2" and EAGLE_LOCAL_PATH.exists():
        return str(EAGLE_LOCAL_PATH)
    return model_path



def _to_dtype(dtype: str) -> torch.dtype:
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32



class EagleObservationEncoder(nn.Module):
    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        freeze_encoder: bool = True,
        trust_remote_code: bool = True,
    ):
        super().__init__()
        self.model_path = _resolve_eagle_path(model_path)
        self.dtype = _to_dtype(dtype)
        self.freeze_encoder = freeze_encoder

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=trust_remote_code,
        )
        self.processor.tokenizer.padding_side = "left"

        # Try pretrained weights first, then config-only fallback for local smoke runs.
        try:
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=trust_remote_code,
                torch_dtype=self.dtype,
            )
        except Exception:
            config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=trust_remote_code)
            self.model = AutoModel.from_config(config, trust_remote_code=trust_remote_code)
            self.model = self.model.to(self.dtype)

        if self.freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

    @property
    def embedding_dim(self) -> int:
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            return 2048
        if hasattr(cfg, "hidden_size"):
            return int(cfg.hidden_size)
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            return int(cfg.text_config.hidden_size)
        return 2048

    def _format_conversations(self, images: list[list[Image.Image]], texts: list[str]) -> tuple[list[str], Any]:
        conversations: list[dict[str, Any]] = []
        for sample_images, sample_text in zip(images, texts, strict=True):
            image_contents = [{"type": "image", "image": img} for img in sample_images]
            conversations.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": sample_text},
                        *image_contents,
                    ],
                }
            )

        text_inputs = [
            self.processor.apply_chat_template(
                [conv],
                tokenize=False,
                add_generation_prompt=False,
            )
            for conv in conversations
        ]

        if hasattr(self.processor, "process_vision_info"):
            image_inputs, _ = self.processor.process_vision_info([[conv] for conv in conversations])
        else:
            image_inputs = [img for sample in images for img in sample]

        return text_inputs, image_inputs

    def encode_batch(self, images: list[list[Image.Image]], texts: list[str], device: torch.device) -> torch.Tensor:
        text_inputs, image_inputs = self._format_conversations(images, texts)
        model_inputs = self.processor(
            text=text_inputs,
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        )
        model_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in model_inputs.items()}

        with torch.set_grad_enabled(not self.freeze_encoder):
            outputs = self.model(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                pixel_values=model_inputs["pixel_values"],
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]
            attention_mask = model_inputs["attention_mask"].bool()
            masked_hidden = hidden * attention_mask.unsqueeze(-1)
            pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
            return pooled


class MockObservationEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def encode_batch(self, images: list[list[Image.Image]], texts: list[str], device: torch.device) -> torch.Tensor:
        batch = len(texts)
        features = torch.zeros(batch, self._embedding_dim, device=device)
        for idx, (sample_images, text) in enumerate(zip(images, texts, strict=True)):
            img_signal = 0.0
            for img in sample_images:
                arr = np.asarray(img, dtype=np.float32)
                img_signal += float(arr.mean() / 255.0)
            text_signal = float(len(text) % 97) / 97.0
            features[idx, 0] = img_signal
            features[idx, 1] = text_signal
            features[idx, 2] = float(len(sample_images))
        return features



def build_encoder(
    backend: str,
    model_path: str,
    dtype: str,
    freeze_encoder: bool,
) -> nn.Module:
    if backend == "mock":
        return MockObservationEncoder()
    return EagleObservationEncoder(
        model_path=model_path,
        dtype=dtype,
        freeze_encoder=freeze_encoder,
    )
