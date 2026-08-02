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
    by the eval harness.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ):
        """
        Initialize the inference wrapper.

        Args:
            checkpoint_path: Path to trained model checkpoint directory.
            device: Device to run inference on.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = greedy).
        """
        self.checkpoint_path = checkpoint_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        print(f"Loading MSTM checkpoint from: {checkpoint_path}")
        self.model = MSTMModel.from_pretrained(
            checkpoint_path,
            device=device,
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
        Batch transition — processes sequentially (generation is serial).

        Args:
            memories: List of current memory states.
            deltas: List of new information (same length).
            **kwargs: Additional generation kwargs.

        Returns:
            List of evolved memory states.
        """
        assert len(memories) == len(deltas)
        return [
            self.transition(m, d, **kwargs)
            for m, d in zip(memories, deltas)
        ]