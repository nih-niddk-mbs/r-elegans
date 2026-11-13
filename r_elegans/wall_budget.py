"""Wall-clock limits are separate from physical simulation time."""
import time


class WallDeadline(RuntimeError):
    pass


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise WallDeadline('Wall-clock budget exhausted; physical simulation time was not rescaled')


def validation_reserve(parent_seconds):
    if parent_seconds <= 0:
        raise ValueError('Measured parent validation time required')
    return max(45*60.,2*parent_seconds+10*60.)


def can_search(remaining,reserve,slowest_evaluation):
    return remaining > reserve+max(10*60.,2*slowest_evaluation)+60
