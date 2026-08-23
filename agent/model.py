from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


class ModelClient(Protocol):
    """
    Interface expected by the agent loop.
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

    Responsible only for model inference.
    """

    DEFAULT_MODEL = "Qwen/Qwen3-8B"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        load_in_4bit: bool = True,
        device_map: str = "cuda:0",
    ) -> None:

        self.model_name = model_name
        self.load_in_4bit = load_in_4bit

        # ---------------------------------------------------------
        # CUDA CHECK
        # ---------------------------------------------------------

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for Qwen3-8B, "
                "but no CUDA GPU is available."
            )

        self.cuda_device = torch.device("cuda:0")

        print("=" * 60)
        print("QWEN MODEL INITIALIZATION")
        print("=" * 60)
        print("Model:", self.model_name)
        print("CUDA:", torch.cuda.get_device_name(0))
        print(
            "GPU memory:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3),
                2,
            ),
            "GB",
        )

        # ---------------------------------------------------------
        # TOKENIZER
        # ---------------------------------------------------------

        self._tokenizer = self._load_tokenizer()

        # ---------------------------------------------------------
        # MODEL
        # ---------------------------------------------------------

        self._model = self._load_model(
            load_in_4bit=load_in_4bit,
            device_map=device_map,
        )

        # ---------------------------------------------------------
        # GENERATION CONFIGS
        # ---------------------------------------------------------

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

        print("QWEN MODEL LOADED")
        print("DEVICE:", self.device)
        print("GPU MEMORY:", self.gpu_memory())
        print("=" * 60)

    # =============================================================
    # TOKENIZER
    # =============================================================

    def _load_tokenizer(self):
        """Load tokenizer from Hugging Face."""

        return AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

    # =============================================================
    # MODEL LOADING
    # =============================================================

    def _load_model(
        self,
        load_in_4bit: bool,
        device_map: str,
    ):
        """
        Load Qwen3-8B.

        For 4-bit inference we explicitly target cuda:0.

        We intentionally do NOT use device_map="auto" because
        Transformers may decide to dispatch some modules to CPU,
        which BitsAndBytes rejects for this configuration.
        """

        compute_dtype = torch.float16

        quantization_config = None

        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )

            effective_device_map = "cuda:0"

        else:
            effective_device_map = device_map

        print()
        print("STARTING QWEN LOAD...")
        print("4-bit:", load_in_4bit)
        print("Device map:", effective_device_map)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map=effective_device_map,
            dtype=compute_dtype,
            trust_remote_code=True,
        )

        model.eval()

        return model

    # =============================================================
    # GENERATION
    # =============================================================

    def generate(
        self,
        messages: list[dict[str, str]],
        mode: str = "tool_selection",
    ) -> str:
        """
        Generate a response from Qwen.
        """

        if mode not in self.generation_configs:
            raise ValueError(
                f"Unknown generation mode: {mode}"
            )

        config = self.generation_configs[mode]

        # ---------------------------------------------------------
        # CHAT TEMPLATE
        # ---------------------------------------------------------

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config.enable_thinking,
        )

        # ---------------------------------------------------------
        # TOKENIZE
        # ---------------------------------------------------------

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
        )

        # ---------------------------------------------------------
        # MOVE INPUTS TO CUDA
        # ---------------------------------------------------------

        inputs = {
            key: value.to(self.cuda_device)
            for key, value in inputs.items()
        }

        # ---------------------------------------------------------
        # GENERATION PARAMETERS
        # ---------------------------------------------------------

        generation_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }

        if config.do_sample:
            generation_kwargs["temperature"] = (
                config.temperature
            )

        # ---------------------------------------------------------
        # GENERATE
        # ---------------------------------------------------------

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                **generation_kwargs,
            )

        # ---------------------------------------------------------
        # REMOVE PROMPT TOKENS
        # ---------------------------------------------------------

        input_length = inputs["input_ids"].shape[-1]

        generated_tokens = outputs[
            0,
            input_length:
        ]

        # ---------------------------------------------------------
        # DECODE
        # ---------------------------------------------------------

        return self._tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

    # =============================================================
    # PROPERTIES
    # =============================================================

    @property
    def model(self):
        """Return underlying Transformers model."""

        return self._model

    @property
    def tokenizer(self):
        """Return tokenizer."""

        return self._tokenizer

    @property
    def device(self):
        """Return model device."""

        return self._model.device

    # =============================================================
    # GPU MEMORY
    # =============================================================

    def gpu_memory(self) -> dict[str, float]:
        """Return current GPU memory usage."""

        if not torch.cuda.is_available():
            return {
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
            }

        return {
            "allocated_gb": round(
                torch.cuda.memory_allocated()
                / (1024 ** 3),
                2,
            ),
            "reserved_gb": round(
                torch.cuda.memory_reserved()
                / (1024 ** 3),
                2,
            ),
        }
