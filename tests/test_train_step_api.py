import jax.numpy as jnp
import ml_collections

from jaxpi.models import MPINN, MPINNSingleChart


def _config():
    cfg = ml_collections.ConfigDict()
    cfg.seed = 0
    cfg.input_dim = 2
    cfg.arch = ml_collections.ConfigDict()
    cfg.arch.arch_name = "Mlp"
    cfg.arch.num_layers = 1
    cfg.arch.hidden_dim = 4
    cfg.arch.out_dim = 1
    cfg.arch.activation = "tanh"
    cfg.arch.periodicity = None
    cfg.arch.fourier_emb = None
    cfg.arch.reparam = None
    cfg.optim = ml_collections.ConfigDict()
    cfg.optim.grad_accum_steps = 1
    cfg.optim.optimizer = "Adam"
    cfg.optim.learning_rate = 1e-3
    cfg.optim.decay_steps = 10
    cfg.optim.decay_rate = 0.9
    cfg.optim.beta1 = 0.9
    cfg.optim.beta2 = 0.999
    cfg.optim.eps = 1e-8
    cfg.optim.lbfgs_learning_rate = 1e-3
    cfg.weighting = ml_collections.ConfigDict()
    cfg.weighting.scheme = "fixed"
    cfg.weighting.init_weights = ml_collections.ConfigDict({"dummy": 1.0})
    cfg.weighting.momentum = 0.9
    return cfg


class DummyMulti(MPINN):
    def u_net(self, params, *args):
        return 0.0

    def r_net(self, params, *args):
        return 0.0

    def losses(self, params, batch, *args):
        return {"dummy": jnp.mean(batch**2)}

    def compute_diag_ntk(self, params, batch, *args):
        return {}

    def compute_l2_error(self, params, eval_x, eval_y, u_eval):
        return 0.0

    def create_losses(self):
        return None


class DummySingle(MPINNSingleChart):
    def u_net(self, params, *args):
        return 0.0

    def r_net(self, params, *args):
        return 0.0

    def losses(self, params, batch, *args):
        return {"dummy": jnp.mean(batch**2)}

    def compute_diag_ntk(self, params, batch, *args):
        return {}

    def compute_l2_error(self, params, eval_x, eval_y, u_eval):
        return 0.0

    def create_losses(self):
        return None


def test_multi_chart_step_returns_loss_aux_state():
    model = DummyMulti(_config(), num_charts=2)
    loss, aux, state = model.step(model.state, jnp.ones((2, 2)))
    assert loss.shape == ()
    assert "losses" in aux
    assert "dummy" in aux
    assert state.step == model.state.step + 1


def test_single_chart_step_returns_loss_aux_state():
    model = DummySingle(_config())
    loss, aux, state = model.step(model.state, jnp.ones((2, 2)))
    assert loss.shape == ()
    assert "losses" in aux
    assert "dummy" in aux
    assert state.step == model.state.step + 1
