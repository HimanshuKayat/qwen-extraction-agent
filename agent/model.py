from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


class ModelClient(Protocol):
    """
    Interface expected by the agent loop.

    Any model implementation used by the agent must expose
    a generate() method with this interface.
    """

    def generate(
        self,
        messages: list[dict[str, str]],
        mode: str = "tool_selection",
    ) -> str:
        ...


@dataclass
class GenerationConfig:
    """
    Configuration for a model generation mode.
    """

    enable_thinking: bool = False
    max_new_tokens: int = 256
    temperature: float = 0.0
    do_sample: bool = False


class QwenModel:
    """
    Qwen3-8B model implementation.

    This class is responsible ONLY for model inference.

    It does not:
    - execute tools
    - download files
    - scrape websites
    - validate datasets
    - access databases
    """

    DEFAULT_MODEL = "Qwen/Qwen3-8B"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        load_in_4bit: bool = True,
        device_map: str = "auto",
    ) -> None:

        self.model_name = model_name

        self._tokenizer = self._load_tokenizer()

        self._model = self._load_model(
            load_in_4bit=load_in_4bit,
            device_map=device_map,
        )

        self.generation_configs = {
            "tool_selection": GenerationConfig(
                enable_thinking=False,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
            ),
            "reasoning": GenerationConfig(
                enable_thinking=True,
                max_new_tokens=1024,
                temperature=0.6,
                do_sample=True,
            ),
        }

    def _load_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

    def _load_model(
        self,
        load_in_4bit: bool,
        device_map: str,
    ):
        quantization_config = None

        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        model.eval()

        return model

    def generate(
        self,
        messages: list[dict[str, str]],
        mode: str = "tool_selection",
    ) -> str:

        if mode not in self.generation_configs:
            raise ValueError(
                f"Unknown generation mode: {mode}"
            )

        config = self.generation_configs[mode]

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config.enable_thinking,
        )

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self._model.device)
            for key, value in inputs.items()
        }

        generation_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }

        if config.do_sample:
            generation_kwargs["temperature"] = config.temperature

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                **generation_kwargs,
            )

        input_length = inputs["input_ids"].shape[-1]

        generated_tokens = outputs[
            0,
            input_length:
        ]

        return self._tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def device(self):
        return self._model.device

    def gpu_memory(self) -> dict[str, float]:
        if not torch.cuda.is_available():
            return {
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
            }

        return {
            "allocated_gb": round(
                torch.cuda.memory_allocated() / (1024 ** 3),
                2,
            ),
            "reserved_gb": round(
                torch.cuda.memory_reserved() / (1024 ** 3),
                2,
            ),
        }