# ABOUTME: LLM SDK for local model inference using Hugging Face transformers.
# ABOUTME: Provides Small_LLM_Model class for loading and running causal language models.

import torch
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    logging,
)

logging.set_verbosity_error()


class Small_LLM_Model:
    """Small wrapper around a Hugging Face causal language model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self._model_name = model_name

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self._device = device

        if dtype is None:
            dtype = torch.float16 if self._device in {"cuda", "mps"} else torch.float32
        self._dtype = dtype

        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=trust_remote_code,
        )
        if self._device != "cuda":
            self._model.to(self._device)
        self._model.eval()

        for param in self._model.parameters():
            param.requires_grad = False

    def encode(self, text: str) -> torch.Tensor:
        """Tokenize text and return a 2-D input_ids tensor."""
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], device=self._device, dtype=torch.long)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        """Decode token ids back to text."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return next-token logits for input_ids.

        This intentionally recomputes the prompt instead of using a fragile cache.
        Constrained decoding changes context often, and wrong cache reuse gives
        wrong logits.
        """
        if not input_ids:
            raise ValueError("input_ids must not be empty")

        input_tensor = torch.tensor([input_ids], device=self._device, dtype=torch.long)
        with torch.no_grad():
            output = self._model(input_ids=input_tensor)
        return [float(x) for x in output.logits[0, -1].tolist()]

    def get_path_to_vocab_file(self) -> str:
        """Return the local path to the tokenizer vocabulary file."""
        vocab_file_name = self._tokenizer.vocab_files_names.get(
            "vocab_file", "vocab.json"
        )
        return hf_hub_download(repo_id=self._model_name, filename=vocab_file_name)

    def get_path_to_merges_file(self) -> str:
        """Return the local path to the tokenizer merges file."""
        merges_file_name = self._tokenizer.vocab_files_names.get(
            "merges_file", "merges.txt"
        )
        return hf_hub_download(repo_id=self._model_name, filename=merges_file_name)

    def get_path_to_tokenizer_file(self) -> str:
        """Return the local path to the tokenizer JSON file."""
        tokenizer_file_name = self._tokenizer.vocab_files_names.get(
            "tokenizer_file", "tokenizer.json"
        )
        return hf_hub_download(repo_id=self._model_name, filename=tokenizer_file_name)
