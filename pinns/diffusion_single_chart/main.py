import os

from absl import app, flags
from ml_collections import config_flags

# Deterministic
# os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_reductions --xla_gpu_autotune_level=0"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # DETERMINISTIC

# Default to CPU execution unless the caller explicitly requests GPU.
target_platform = os.environ.get("JAX_PLATFORM_NAME")
if target_platform is None:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    target_platform = "cpu"

if target_platform == "gpu":
    os.environ.setdefault(
        "XLA_FLAGS",
        " ".join(
            [
                "--xla_gpu_enable_triton_softmax_fusion=true",
                "--xla_gpu_triton_gemm_any=false",
                "--xla_gpu_enable_async_collectives=true",
                "--xla_gpu_enable_latency_hiding_scheduler=true",
                "--xla_gpu_enable_highest_priority_async_stream=true",
            ]
        ),
    )
else:
    os.environ.pop("XLA_FLAGS", None)

FLAGS = flags.FLAGS


config_flags.DEFINE_config_file(
    "config",
    default="./configs/default.py",
    help_string="File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    """Route execution to training, evaluation or data generation."""
    if FLAGS.config.mode == "train":
        from . import train

        train.train_and_evaluate(FLAGS.config)

    elif FLAGS.config.mode == "eval":
        from . import eval

        eval.evaluate(FLAGS.config)

    elif FLAGS.config.mode == "generate_data":
        from . import generate_data

        generate_data.generate_data(FLAGS.config)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config"])
    app.run(main)
