from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any

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
        device_map: str | dict[str, Any] = "auto",
    ) -> None:

        self.model_name = model_name
        self.load_in_4bit = load_in_4bit

        # ---------------------------------------------------------
        # Environment validation
        # ---------------------------------------------------------

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for Qwen3-8B in the current "
                "configuration, but no CUDA GPU is available."
            )

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

    # =============================================================
    # TOKENIZER
    # =============================================================

    def _load_tokenizer(self):
        """Load the Qwen tokenizer."""

        return AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

    # =============================================================
    # MODEL
    # =============================================================

    def _load_model(
        self,
        load_in_4bit: bool,
        device_map: str | dict[str, Any],
    ):
        """
        Load Qwen.

        When using BitsAndBytes 4-bit quantization, we deliberately
        place the model on the CUDA device rather than allowing
        Accelerate's automatic device mapping to partially dispatch
        modules to CPU.

        This avoids the error:

            Some modules are dispatched on the CPU or the disk.

        which occurs when a quantized model receives a mixed
        GPU/CPU device map without the required offload settings.
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

            # -----------------------------------------------------
            # IMPORTANT
            #
            # Do NOT let device_map="auto" create:
            #
            #     GPU + CPU
            #
            # for a 4-bit BitsAndBytes model.
            #
            # A T4 should load Qwen3-8B 4-bit entirely on GPU.
            # -----------------------------------------------------

            effective_device_map = {
                "": 0,
            }

        else:
            # For non-quantized loading we can retain the caller's
            # requested device map.
            effective_device_map = device_map

        print(
            f"[QwenModel] Loading model: {self.model_name}"
        )

        print(
            f"[QwenModel] 4-bit quantization: {load_in_4bit}"
        )

        print(
            f"[QwenModel] Device map: {effective_device_map}"
        )

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
        Generate a model response.

        Args:
            messages:
                Chat messages in standard role/content format.

            mode:
                Generation configuration to use.

        Returns:
            Raw generated model text.
        """

        if mode not in self.generation_configs:
            raise ValueError(
                f"Unknown generation mode: {mode}"
            )

        config = self.generation_configs[mode]

        # ---------------------------------------------------------
        # Build Qwen chat prompt
        # ---------------------------------------------------------

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config.enable_thinking,
        )

        # ---------------------------------------------------------
        # Tokenize
        # ---------------------------------------------------------

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
        )

        # ---------------------------------------------------------
        # Move inputs to model device
        # ---------------------------------------------------------

        model_device = self._model.device

        inputs = {
            key: value.to(model_device)
            for key, value in inputs.items()
        }

        # ---------------------------------------------------------
        # Generation configuration
        # ---------------------------------------------------------

        generation_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }

        if config.do_sample:
            generation_kwargs["temperature"] = config.temperature

        # ---------------------------------------------------------
        # Generate
        # ---------------------------------------------------------

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                **generation_kwargs,
            )

        # ---------------------------------------------------------
        # Remove input tokens
        # ---------------------------------------------------------

        input_length = inputs["input_ids"].shape[-1]

        generated_tokens = outputs[
            0,
            input_length:
        ]

        # ---------------------------------------------------------
        # Decode
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
        """Return the underlying Transformers model."""

        return self._model

    @property
    def tokenizer(self):
        """Return the tokenizer."""

        return self._tokenizer

    @property
    def device(self):
        """Return the model device."""

        return self._model.device

    # =============================================================
    # GPU MEMORY
    # =============================================================

    def gpu_memory(self) -> dict[str, float]:
        """Return current GPU memory usage in GB."""

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
