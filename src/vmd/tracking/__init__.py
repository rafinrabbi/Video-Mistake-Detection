"""Online procedural-state tracking over a task graph.

``belief`` holds the DAG-constrained filter that replaces the flat step pointer;
``replay`` re-runs any tracker over saved per-window records on the CPU.
"""
from vmd.tracking.belief import (  # noqa: F401
    BeliefFilter, BeliefState, FilterConfig, StepSpace, build_step_space,
)
