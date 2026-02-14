"""
Numerical Stability Layer - Standard ISO/IEEE for Mission Critical AI.
Prevents division by zero, NaNs, and Infs during high-dimensional audio processing.
"""
import torch
import logging

class NumericalGuard:
    @staticmethod
    def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """
        Performs division with epsilon padding to prevent NaN results.
        Formula: x / (y + eps)
        """
        return numerator / (denominator + eps)

    @staticmethod
    def safe_log(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """Computes stable log to prevent -inf in spectral domains."""
        return torch.log(x + eps)

    @staticmethod
    def validate_tensor(tensor: torch.Tensor, name: str):
        """Standard DARPA check for tensor integrity."""
        if not torch.isfinite(tensor).all():
            logging.error(f"CRITICAL: Numerical instability in {name}")
            raise ArithmeticError(f"System Lock Prevention: Non-finite values in {name}")
        return True

    @staticmethod
    def shape_assertion(tensor: torch.Tensor, expected_shape: tuple, context: str):
        """Strict shape checking to prevent broadcasting errors."""
        if tensor.shape != expected_shape:
            raise TypeError(f"Data Structure Mismatch in {context}: Got {tensor.shape}, expected {expected_shape}")
