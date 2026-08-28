"""BCRD three-gate falsification harness.

CPU replay is a protocol/code check only.  Scientific decisions require native
route traces and CUDA service curves accepted by the individual entrypoints.
"""

from .core import Contribution, ProtocolError, ServiceCatalog

__all__ = ["Contribution", "ProtocolError", "ServiceCatalog"]
