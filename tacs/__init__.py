"""Trusted-Anchor Curriculum Self-Training (TACS) for protein classifiers.

The package keeps weak TrEMBL supervision separate from trusted Swiss-Prot
supervision.  It is intentionally designed around frozen ESM-2 embeddings so
that expensive encoder inference can be cached and reused across curriculum
rounds and ablations.
"""

from .config import load_config

__all__ = ["load_config"]
