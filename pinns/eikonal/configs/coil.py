from datetime import datetime
import ml_collections
from manifold_pinns.pipeline.env import (
    batch_path,
    checkpoint_path,
    data_path,
    env_bool,
    env_int,
    env_path,
    eval_path,
    figure_path,
    profiler_path,
    run_path,
    wandb_entity,
    wandb_project,
)


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.figure_path = figure_path(
        "eikonal",
        "coil",
        str(datetime.now().strftime("%Y%m%d-%H%M%S")),
    )

    config.plot = False

    config.runtime = ml_collections.ConfigDict()
    config.runtime.enable_x64 = False

    config.mode = "train"
    config.N = 20  # Number of gt points in the training set
    config.idxs = None
    config.bcs_seed = 42

    config.num_supernodes = 128

    config.chart = ml_collections.ConfigDict()
    config.chart.backend = "uae"

    config.dataset = ml_collections.ConfigDict()
    config.dataset.charts_path = env_path(
        "MANIFOLD_PINNS_EIKONAL_COIL_CHARTS_PATH",
        data_path("coil", "charts_1"),
    )
    config.dataset.regenerate_charts2d = False

    config.eikonal = ml_collections.ConfigDict()
    config.eikonal.enforce_source_bc = True
    config.eikonal.source_idx = 0
    config.eikonal.source_bc_weight = 1.0
    config.eikonal.hard_source_ansatz = False

    config.sparse_points = ml_collections.ConfigDict()
    config.sparse_points.path = env_path(
        "MANIFOLD_PINNS_EIKONAL_COIL_SPARSE_POINTS_PATH",
        run_path("sparse_points", "eikonal", "coil"),
    )
    config.sparse_points.strategy = "random"
    config.sparse_points.num_bins = None

    # Autoencoder checkpoint
    config.autoencoder_checkpoint = ml_collections.ConfigDict()
    config.autoencoder_checkpoint.checkpoint_path = env_path(
        "MANIFOLD_PINNS_UAE_COIL_CHECKPOINT",
        checkpoint_path("uae", "coil", "latest"),
    )
    config.autoencoder_checkpoint.step = env_int("MANIFOLD_PINNS_UAE_COIL_STEP", 0)

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.use = env_bool("MANIFOLD_PINNS_WANDB_USE", False)
    wandb.project = wandb_project("M-PINN")
    wandb.entity = wandb_entity()
    wandb.name = "default"
    wandb.tag = None
    wandb.log_every_steps = 100
    wandb.eval_every_steps = 100

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    arch.num_layers = 1
    arch.hidden_dim = 16
    arch.out_dim = 1
    arch.activation = "tanh"

    # arch.periodicity = ml_collections.ConfigDict(
    #     {"period": (jnp.pi,), "axis": (1,), "trainable": (False,)}
    # )

    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1, "embed_dim": 32})
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 0.5, "stddev": 0.1}
    )

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.grad_accum_steps = 0
    optim.optimizer = "Adam"  #  "AdamWarmupCosineDecay"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3

    optim.lbfgs_learning_rate = 0.00001
    optim.decay_rate = 0.9

    # cosine decay
    optim.warmup_steps = 1000
    optim.decay_steps = 10000

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 100000
    training.batch_size = 128  # 1024
    training.lbfgs_max_steps = 0

    training.load_existing_batches = True
    training.batches_path = batch_path("eikonal", "coil")
    training.num_boundary_batches = 500

    # training.res_batches_path = "pinns/eikonal/coil/data/res_batches.npy"
    # training.boundary_batches_path = (
    #     "pinns/eikonal/coil/data/boundary_batches.npy"
    # )
    # training.boundary_pairs_idxs_path = (
    #     "pinns/eikonal/coil/data/boundary_pairs_idxs.npy"
    # )
    # training.bcs_batches_path = "pinns/eikonal/coil/data/bcs_batches.npy"
    # training.bcs_values_path = "pinns/eikonal/coil/data/bcs_values.npy"

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict(
        {"bcs": 1.0, "res": 1.0, "bc": 1.0, "source": 1.0}
    )
    weighting.momentum = 0.9
    weighting.update_every_steps = 100000

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.eval_every_steps = 1000
    logging.num_eval_points = 5000

    logging.log_errors = False
    logging.log_losses = True
    logging.log_weights = False
    logging.log_preds = False
    logging.log_grads = False
    logging.log_ntk = False

    config.profiler = profiler = ml_collections.ConfigDict()
    profiler.start_step = 200
    profiler.end_step = 210
    profiler.log_dir = profiler_path("eikonal", "coil")

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.checkpoint_dir = checkpoint_path("pinn", "eikonal", "coil")
    saving.save_every_steps = 100000
    saving.num_keep_ckpts = 5

    # Eval
    config.eval = eval = ml_collections.ConfigDict()
    eval.eval_with_last_ckpt = False
    eval.checkpoint_dir = env_path(
        "MANIFOLD_PINNS_EIKONAL_COIL_CHECKPOINT",
        saving.checkpoint_dir,
    )
    eval.step = 100000
    eval.N = 2000
    eval.use_existing_solution = False
    eval.solution_path = eval_path("eikonal", "coil")
    eval.plot_everything = True

    # Input shape for initializing Flax models
    config.input_dim = 2

    # Integer for PRNG random seed.
    config.seed = 42

    return config
