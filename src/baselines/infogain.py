"""Dataset-generic classification adaptation of Document Information Gain.

The compatibility module :mod:`src.baselines.infogain_fever` remains available
for frozen FEVER manifests and imports.  The implementation is dataset-neutral:
it consumes only the shared posterior and candidate schemas.
"""

from src.baselines.infogain_fever import (  # noqa: F401
    HELD_OUT_ROLES,
    TEACHER_DEFINITION,
    TEACHER_PURPOSES,
    TRAINING_ROLES,
    VALIDATION_ROLES,
    group_teacher_rows,
    infogain_multitask_loss,
    label_dig,
    pointwise_input,
    posterior_to_teacher_rows,
    resolve_thresholds,
    validate_probability_vector,
    validate_teacher_roles,
    validate_teacher_rows_for_training,
)

__all__ = [name for name in globals() if not name.startswith("_")]
