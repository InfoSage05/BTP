"""
physics
-------
Closed-form CHF physics for the unified pipeline: saturation properties,
literature correlations, the dimensionless feature map, the regime-dispatched
baseline scale, and the physics-consistency scorecard.

Every constant traces to a source catalogued in
`physics_foundation/CHF_Physics_Foundation.md`; nothing here is fitted.
"""
from . import baseline, constraints, correlations, groups, properties  # noqa: F401

__all__ = ["baseline", "constraints", "correlations", "groups", "properties"]
