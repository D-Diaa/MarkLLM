import os
from dataclasses import dataclass
from typing import Dict
from typing import Tuple

import torch
from peft import PeftModel, PeftConfig
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelConfig:
    """Configuration for model loading and inference."""
    model_name: str
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_new_tokens: int = 256
    do_sample: bool = True


def _load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load tokenizer with error handling."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        tokenizer.padding_side = "left"
        return tokenizer
    except Exception as e:
        raise ValueError(f"Failed to load tokenizer {model_name}.") from e


def _interpolate_weights(base_model: nn.Module, adapted_model: nn.Module, ratio: float):
    """Interpolate between base model and adapted model weights."""
    with torch.no_grad():
        for name, param in base_model.named_parameters():
            if name in adapted_model.state_dict():
                adapted_param = adapted_model.state_dict()[name]
                # Interpolate: new_weight = (1-ratio)*base + ratio*adapted
                interpolated_param = (1 - ratio) * param + ratio * adapted_param
                param.copy_(interpolated_param)


class ModelLoader:
    """Handles loading and configuration of language models with optional adapter ratio."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load(self, adapter_ratio: float = 1.0) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model and tokenizer based on configuration.

        Args:
            adapter_ratio: Float between 0 and 1 indicating how much to apply the adapter.
                         0 = pure base model
                         1 = fully adapted model
                         values between 0-1 = interpolated model
        """
        if not 0 <= adapter_ratio <= 1:
            raise ValueError("adapter_ratio must be between 0 and 1")

        if self._is_adapter_model():
            print(f"Loading model with PEFT adapter (ratio: {adapter_ratio})")
            self._load_adapter_model(adapter_ratio)
        else:
            print("Loading base model")
            self._load_base_model()

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        return self.model, self.tokenizer

    def _load_adapter_model(self, ratio: float = 1.0):
        """Load a model with PEFT adapter, optionally interpolating between base and adapted weights."""
        # Load base model first
        peft_config = PeftConfig.from_pretrained(self.config.model_name)
        base_model = self._load_model(peft_config.base_model_name_or_path)

        if ratio == 1.0:
            # Standard adapter loading if ratio is 1
            self.model = PeftModel.from_pretrained(base_model, self.config.model_name)
        elif ratio == 0.0:
            # Load base model only
            self.model = base_model
        else:
            # Load adapter model separately
            adapted_model = PeftModel.from_pretrained(base_model, self.config.model_name)
            adapted_model = adapted_model.merge_and_unload(safe_merge=True, progressbar=True)
            base_model = self._load_model(peft_config.base_model_name_or_path)
            # Interpolate between base and adapted weights
            _interpolate_weights(base_model, adapted_model, ratio)
            del adapted_model
            self.model = base_model
        torch.cuda.empty_cache()
        self.tokenizer = _load_tokenizer(peft_config.base_model_name_or_path)

    @classmethod
    def from_name(cls,
                  model_name: str,
                  device: str = None,
                  adapter_ratio: float = 1.0) -> Dict:
        """Create a model loader from model name and optional device."""
        config = ModelConfig(model_name=model_name)
        if device:
            config.device = device
        model, tokenizer = cls(config).load(adapter_ratio=adapter_ratio)
        return {'model': model, 'tokenizer': tokenizer}

    def _is_adapter_model(self) -> bool:
        """Check if the model path contains an adapter configuration."""
        model_path = self.config.model_name
        adapter_config_path = os.path.join(model_path, 'adapter_config.json')
        return (os.path.exists(model_path) and
                not os.path.exists(os.path.join(model_path, 'config.json')) and
                os.path.exists(adapter_config_path))

    def _load_base_model(self):
        """Load a base model without adapters."""
        self.model = self._load_model(self.config.model_name)
        self.tokenizer = _load_tokenizer(self.config.model_name)

    def _load_model(self, model_name: str) -> AutoModelForCausalLM:
        """Load model with error handling."""
        try:
            return AutoModelForCausalLM.from_pretrained(model_name,
                                                        torch_dtype=torch.bfloat16,
                                                        ).to(self.config.device)
        except Exception as e:
            raise ValueError(f"Failed to load model {model_name}.") from e
