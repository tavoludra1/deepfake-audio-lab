"""
Topological Signature Engine for Deepfake Detection.
Implements Takens Embedding and Persistent Homology.
"""
import numpy as np
from gtda.homology import VietorisRipsPersistence
from gtda.time_series import TakensEmbedding

class SonicSignature:
    def __init__(self, dimension: int = 3, delay: int = 1):
        # Time delay embedding parameters
        self.embedder = TakensEmbedding(dimension=dimension, time_delay=delay)
        # Homology: 0 (connected components), 1 (loops/holes)
        self.homology = VietorisRipsPersistence(homology_dimensions=[0, 1])

    def generate_fingerprint(self, signal: np.ndarray) -> np.ndarray:
        """
        Extracts the topological manifold of the audio signal.
        The hypothesis is that Deepfakes exhibit 'broken' topological manifolds.
        """
        # CRITICAL FIX: GTDA expects shape (n_samples, n_time_steps)
        # If signal is (N,), we reshape to (1, N) representing one time series.
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)
        
        # Phase space reconstruction
        # Output shape: (n_samples, n_points, dimension)
        embedded = self.embedder.fit_transform(signal)
        
        # Compute persistent diagrams (H0, H1)
        # Output shape: (n_samples, n_features, 3)
        diagrams = self.homology.fit_transform(embedded)
        return diagrams
