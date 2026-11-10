"""CRM: Concomitance-guided occlusion compensation with synapse-aware reasoning.

Reproduction of the model described in "CRM 11.6" (Sections III-A..III-E).
The module layout follows the paper's four operations:

    x -> COC -> SSA -> STM -> SIR -> (logits, confidence, consistency)

Every class names the equations it implements so the code can be diffed
against the manuscript.
"""

from .regions import AU_REGIONS, region_masks, assign_regions
from .coc import COCModule
from .ssa import SSAModule, sinkhorn
from .stm import TokenBuilder
from .sir import SIRStack
from .model import CRM
from .losses import CRMLoss

__all__ = [
    "AU_REGIONS",
    "region_masks",
    "assign_regions",
    "COCModule",
    "SSAModule",
    "sinkhorn",
    "TokenBuilder",
    "SIRStack",
    "CRM",
    "CRMLoss",
]
