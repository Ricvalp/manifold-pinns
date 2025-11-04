import os
import jax


def get_last_checkpoint_dir(workdir):
    """Return the most recent run directory containing checkpoints.

    New-style runs are stored directly under ``<workdir>/<wandb_run_id>``.
    Legacy runs used ``<workdir>/<timestamp>/<wandb_run_id>``.  This helper
    discovers both layouts and returns the relative path from ``workdir`` to
    the newest run (based on modification time).
    """

    if not os.path.isdir(workdir):
        raise FileNotFoundError(f"Checkpoint directory '{workdir}' does not exist")

    candidates = []

    def register(path, rel_path):
        # Require either a cfg.json or at least one checkpoint_* folder
        cfg_path = os.path.join(path, "cfg.json")
        has_cfg = os.path.isfile(cfg_path)
        has_ckpt = any(
            child.is_dir() and child.name.startswith("checkpoint_")
            for child in os.scandir(path)
        )
        if has_cfg or has_ckpt:
            candidates.append((os.path.getmtime(path), rel_path))

    with os.scandir(workdir) as entries:
        for entry in entries:
            if not entry.is_dir():
                continue
            # First try treating this directory as a run directory.
            register(entry.path, entry.name)

            # Backward compatibility: check nested directories.
            with os.scandir(entry.path) as sub_entries:
                for sub_entry in sub_entries:
                    if sub_entry.is_dir():
                        rel_path = os.path.join(entry.name, sub_entry.name)
                        register(sub_entry.path, rel_path)

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found under '{workdir}'. "
            "Ensure training has produced at least one run."
        )

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def set_profiler(profiler_config, step, log_dir):
    # Profiling.
    if profiler_config is not None:
        if step == profiler_config.start_step:
            jax.profiler.start_trace(log_dir=log_dir)
        if step == profiler_config.end_step:
            jax.profiler.stop_trace()
