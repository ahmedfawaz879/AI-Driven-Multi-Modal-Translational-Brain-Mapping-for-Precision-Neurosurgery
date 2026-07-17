"""Functional connectivity extraction and graph-theory metrics.

Extracted from the original script (ConnectivityAnalyzer). Logic is
unchanged. ``sklearn.covariance.GraphicalLassoCV`` and ``networkx`` are
imported lazily inside methods, matching the original script, since they
are only needed for the 'partial_correlation' and graph-metric code paths
respectively.

Note: the original script also imported ``nilearn.image.resample_to_img``
at module scope but never called it anywhere. That dead import is dropped
here rather than carried forward.
"""

from typing import Dict, Optional

import numpy as np
from nilearn.input_data import NiftiLabelsMasker

from .utils import logger

# ======================== Functional Connectivity ========================


class ConnectivityAnalyzer:
    """Advanced functional connectivity analysis"""

    def __init__(self, atlas_path: str):
        self.atlas_path = atlas_path
        self.masker = NiftiLabelsMasker(labels_img=atlas_path, standardize=True)

    def extract_timeseries(self, fmri_path: str, confounds: Optional[np.ndarray] = None) -> np.ndarray:
        """Extract ROI time series"""
        return self.masker.fit_transform(fmri_path, confounds=confounds)

    def compute_connectivity(self, timeseries: np.ndarray, method: str = "correlation") -> np.ndarray:
        """Compute connectivity matrix"""
        if method == "correlation":
            return np.corrcoef(timeseries.T)
        elif method == "partial_correlation":
            from sklearn.covariance import GraphicalLassoCV

            model = GraphicalLassoCV()
            model.fit(timeseries)
            return model.precision_
        else:
            raise ValueError(f"Unknown method: {method}")

    def network_metrics(self, connectivity: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute graph theory metrics"""
        try:
            import networkx as nx

            # Create graph
            G = nx.from_numpy_array(np.abs(connectivity))

            # Compute metrics
            metrics = {
                "degree": np.array(list(dict(G.degree()).values())),
                "betweenness": np.array(list(nx.betweenness_centrality(G).values())),
                "clustering": np.array(list(nx.clustering(G).values())),
            }

            return metrics
        except ImportError:
            logger.warning("NetworkX not available for graph metrics")
            return {}
