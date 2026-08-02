"""
Inference wrapper for a trained MSTM checkpoint.

Given (M, delta_M), returns M_prime.
Compatible with the eval harness (src/eval/run_eval.py) which expects
a transition(M, delta_M) -> M_prime interface.

Usage:
    from src.mstm.inference import MSTMInference

    infer = MSTMInference("../../experiments/checkpoints/mstm_final")
    M_prime = infer.transition(M, delta_M)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.mstm.model import MSTMModel


class MSTMInference:
    """
    Inference wrapper for a trained MSTM checkpoint.

    Provides the transition(M, delta_M) -> M_prime interface expected
    by the eval harness. Also supports batched inference for high GPU
    utilization.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        batch_size: int = 16,
        compile_model: bool = False,
    ):
        """
        Initialize the inference wrapper.

        Args:
            checkpoint_path: Path to trained model checkpoint directory.
            device: Device to run inference on.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = greedy).
            batch_size: Number of (M, delta_M) pairs to process in one
                        forward pass. Higher = better GPU utilization.
                        Default 16 for 0.6B on 24GB GPU.
            compile_model: Whether to torch.compile the model.
        """
        self.checkpoint_path = checkpoint_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size

        print(f"Loading MSTM checkpoint from: {checkpoint_path}")
        self.model = MSTMModel.from_pretrained(
            checkpoint_path,
            device=device,
            compile_model=compile_model,
        )
        self.model.model.eval()

    def transition(
        self,
        M: str,
        delta_M: str,
        **kwargs,
    ) -> str:
        """
        Generate the evolved memory state.

        Args:
            M: Current memory state as a string.
            delta_M: New information as a string.
            **kwargs: Additional generation kwargs (max_new_tokens, temperature).

        Returns:
            M_prime: The evolved memory state as a string.
        """
        max_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        do_sample = temperature > 0.0

        return self.model.generate(
            M=M,
            delta_M=delta_M,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

    def transition_batch(
        self,
        memories: list,
        deltas: list,
        **kwargs,
    ) -> list:
        """
        Batch transition — processes multiple (M, delta_M) pairs in a
        single batched forward pass for high GPU utilization.

        Args:
            memories: List of current memory states.
            deltas: List of new information (same length).
            **kwargs: Additional generation kwargs.

        Returns:
            List of evolved memory states.
        """
        assert len(memories) == len(deltas), (
            f"memories ({len(memories)}) and deltas ({len(deltas)}) "
            f"must have the same length"
        )

        max_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        do_sample = temperature > 0.0
        batch_size = kwargs.get("batch_size", self.batch_size)

        return self.model.generate_batch(
            M_list=memories,
            delta_M_list=deltas,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=do_sample,
            batch_size=batch_size,
        )