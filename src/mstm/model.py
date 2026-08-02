"""
MSTM — Memory State Transition Model.

Wraps the chosen SLM backbone (Qwen 2.5 0.5B) for the generative transition
T(M, delta_M) -> M_prime.

This is a single generative rewrite, NOT a discrete action classifier.
Do not add an operation-classification head.

Architecture:
- Loads a pre-trained Qwen 2.5 0.5B (or similar) as the base model
- Optionally applies LoRA adapters for parameter-efficient fine-tuning
- Formats input as a structured prompt: [M] + [ΔM] → generate [M′]
- Outputs the full evolved memory state as a single generation

See PROPOSAL.md Section 5, TODOLIST.md Phase 4.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


# ---------------------------------------------------------------------------
# Prompt format for the generative transition
# ---------------------------------------------------------------------------

TRANSITION_PROMPT_TEMPLATE = """<|im_start|>system
You are a memory state transition model. Given the current memory state (M) and new information (ΔM), produce the updated memory state (M′). Update, consolidate, abstract, or forget as needed. Output only the new memory state.<|im_end|>
<|im_start|>user
CURRENT MEMORY (M):
{M}

NEW INFORMATION (ΔM):
{delta_M}

Produce the UPDATED MEMORY STATE (M′):<|im_end|>
<|im_start|>assistant
{M_prime}"""


def format_transition_prompt(
    M: str,
    delta_M: str,
    M_prime: str = "",
    for_training: bool = True,
) -> str:
    """
    Format a (M, delta_M, M_prime) triplet into the training/inference prompt.

    Args:
        M: Current memory state.
        delta_M: New information.
        M_prime: Target evolved memory state (empty for inference).
        for_training: If True, include M_prime for teacher forcing.
                     If False, leave M_prime empty for generation.

    Returns:
        Formatted prompt string.
    """
    if for_training:
        return TRANSITION_PROMPT_TEMPLATE.format(
            M=M.strip(),
            delta_M=delta_M.strip(),
            M_prime=M_prime.strip(),
        )
    else:
        # For inference: stop at the assistant prefix
        prompt = TRANSITION_PROMPT_TEMPLATE.format(
            M=M.strip(),
            delta_M=delta_M.strip(),
            M_prime="",
        )
        # Remove the trailing M_prime (which is empty) and the <|im_end|> won't appear
        return prompt.rstrip()


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class MSTMModel:
    """
    Memory State Transition Model wrapper.

    Loads a pre-trained SLM and provides methods for training and inference
    of the T(M, delta_M) -> M_prime transition.
    """

    def __init__(
        self,
        backbone: str = "Qwen/Qwen3-0.6B",
        device: str = None,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        max_seq_length: int = 2048,
        compile_model: bool = True,
    ):
        """
        Initialize the MSTM model.

        Args:
            backbone: HuggingFace model ID or path.
            device: Device to load model on ('cuda', 'cpu', or None for auto).
            use_lora: Whether to apply LoRA adapters.
            lora_r: LoRA rank.
            lora_alpha: LoRA alpha scaling factor.
            lora_dropout: LoRA dropout rate.
            max_seq_length: Maximum sequence length for tokenization.
            compile_model: Whether to torch.compile the model for faster
                          inference (2-3x speedup for small models on GPU).
        """
        self.backbone_name = backbone
        self.max_seq_length = max_seq_length
        self.use_lora = use_lora
        self._compiled = compile_model

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading tokenizer: {backbone}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            backbone,
            trust_remote_code=True,
            padding_side="right",
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model: {backbone} (device: {self.device})")
        self.model = AutoModelForCausalLM.from_pretrained(
            backbone,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto" if self.device == "cuda" else None,
        )

        if use_lora:
            self._apply_lora(lora_r, lora_alpha, lora_dropout)

        if self.device == "cpu" and not use_lora:
            self.model = self.model.to(self.device)

        # torch.compile for faster inference on GPU
        if compile_model and self.device == "cuda":
            print("Compiling model with torch.compile (mode='reduce-overhead')...")
            try:
                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                print("  Model compiled successfully.")
            except Exception as e:
                print(f"  torch.compile failed ({e}), falling back to eager mode.")
                self._compiled = False

    def _apply_lora(self, r: int, alpha: int, dropout: float):
        """Apply LoRA adapters to the model."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError(
                "peft is required for LoRA. Install with: pip install peft"
            )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def save_pretrained(self, path: str):
        """Save model and tokenizer to disk."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: str = None,
        compile_model: bool = True,
    ):
        """
        Load a trained MSTM checkpoint.

        Args:
            path: Path to the saved model directory.
            device: Device to load on.
            compile_model: Whether to torch.compile the model.

        Returns:
            MSTMModel instance.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        instance = cls.__new__(cls)
        instance.device = device
        instance.max_seq_length = 2048
        instance.use_lora = True  # Assume LoRA was used
        instance._compiled = compile_model

        print(f"Loading tokenizer from: {path}")
        instance.tokenizer = AutoTokenizer.from_pretrained(
            path,
            local_files_only=True,
        )

        if instance.tokenizer.pad_token is None:
            instance.tokenizer.pad_token = instance.tokenizer.eos_token

        print(f"Loading model from: {path}")
        instance.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
            local_files_only=True,
        )

        # torch.compile for faster inference
        if compile_model and device == "cuda":
            print("Compiling model with torch.compile (mode='reduce-overhead')...")
            try:
                instance.model = torch.compile(
                    instance.model,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                print("  Model compiled successfully.")
            except Exception as e:
                print(f"  torch.compile failed ({e}), falling back to eager mode.")
                instance._compiled = False

        return instance

    def generate(
        self,
        M: str,
        delta_M: str,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        do_sample: bool = False,
    ) -> str:
        """
        Generate the evolved memory state M_prime.

        Args:
            M: Current memory state.
            delta_M: New information.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling (False = greedy).

        Returns:
            Generated M_prime as a string.
        """
        prompt = format_transition_prompt(M, delta_M, for_training=False)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part (after the prompt)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        result = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return result.strip()

    @torch.no_grad()
    def generate_batch(
        self,
        M_list: list,
        delta_M_list: list,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        do_sample: bool = False,
        batch_size: int = None,
    ) -> list:
        """
        Generate M_prime for multiple (M, delta_M) pairs in a single batched
        forward pass. This is the key optimization for GPU utilization —
        a 0.6B model on a 24GB GPU can process 16–32 samples simultaneously.

        Args:
            M_list: List of current memory states.
            delta_M_list: List of new information (same length as M_list).
            max_new_tokens: Maximum tokens to generate per sample.
            temperature: Sampling temperature (0.0 = greedy).
            do_sample: Whether to use sampling.
            batch_size: Max samples per forward pass (None = auto from VRAM
                        or process all at once). For a 0.6B model on 24GB,
                        defaults to 16.

        Returns:
            List of generated M_prime strings (same order as input).
        """
        assert len(M_list) == len(delta_M_list), (
            f"M_list ({len(M_list)}) and delta_M_list ({len(delta_M_list)}) "
            f"must have the same length"
        )

        if batch_size is None:
            # Default: 16 is safe for 0.6B on 24GB; auto-scale up for bigger GPUs
            batch_size = 16

        all_results = []

        for batch_start in range(0, len(M_list), batch_size):
            batch_end = min(batch_start + batch_size, len(M_list))
            batch_M = M_list[batch_start:batch_end]
            batch_delta = delta_M_list[batch_start:batch_end]

            # Format all prompts
            prompts = [
                format_transition_prompt(M, delta_M, for_training=False)
                for M, delta_M in zip(batch_M, batch_delta)
            ]

            # Batch-tokenize with padding
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
            ).to(self.device)

            # Batched generation
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # Enable KV-cache for faster autoregressive decoding
                use_cache=True,
            )

            # Decode each sample: only the newly-generated tokens
            input_lens = inputs["attention_mask"].sum(dim=1)  # Real prompt length per sample
            for i in range(len(prompts)):
                sample_ids = outputs[i]
                prompt_len = input_lens[i].item()
                # Remove the prompt prefix (including padding)
                generated_ids = sample_ids[prompt_len:]
                result = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                all_results.append(result.strip())

        return all_results