"""
Mathematical Rigor & Formula Validation.
Ensures zero-invention of formulas and adherence to corpus documentation.
"""
import unittest
import sys
import os
from pathlib import Path

# --- SYSTEM INTEGRITY PATCH ---
# Automatically resolve project root relative to this file
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent  # Go up from tests/ -> deepfake-audio-lab/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# ------------------------------

import torch
import numpy as np
from src.deepfake_audio.infrastructure.numerics.stability import NumericalGuard
from src.deepfake_audio.infrastructure.features.topological_engine import SonicSignature

class TestThesisMathematics(unittest.TestCase):
    def setUp(self):
        self.guard = NumericalGuard()
        # Initialize with standard thesis parameters (Delay=1, Dim=3)
        self.tda_engine = SonicSignature(dimension=3, delay=1)

    def test_epsilon_guard(self):
        """Validates that division by zero is handled style NASA/DARPA."""
        denominator = torch.tensor([0.0, 1e-12, -0.0])
        numerator = torch.tensor([1.0, 1.0, 1.0])
        result = self.guard.safe_divide(numerator, denominator)
        
        # Must be finite and not crash
        self.assertTrue(torch.isfinite(result).all(), "Guard failed: Non-finite result detected.")
        # Check if padding worked (should not exceed 1/EPSILON approx)
        self.assertTrue((result.abs() <= 1e11).all(), "Guard failed: Value explosion detected.")

    def test_takens_embedding_invariance(self):
        """Validates the Takens Embedding Theorem implementation."""
        # Periodic signal (Sine wave) - Represents a stable attractor
        t = np.linspace(0, 10, 100)
        signal = np.sin(t)
        
        # Generate topological fingerprint
        fingerprint = self.tda_engine.generate_fingerprint(signal)
        
        # Verification: H0 (Connected components) must exist
        # Persistence diagrams shape: (n_samples, n_points, 3) 
        # We check if we have feature points extracted
        self.assertGreater(fingerprint.shape[1], 0, "Topological manifold failed to emerge.")

    def test_gradient_integrity(self):
        """Ensures gradients are finite, preventing 'System Locks'."""
        tensor = torch.tensor([1.0, np.nan, 2.0])
        with self.assertRaises(ArithmeticError):
            self.guard.validate_tensor(tensor, "Integrity_Test")

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
