from .metrics import compute_all_metrics


def run_sweep(*args, **kwargs):
    from .sweep import run_sweep as _run_sweep

    return _run_sweep(*args, **kwargs)


__all__ = ["compute_all_metrics", "run_sweep"]
