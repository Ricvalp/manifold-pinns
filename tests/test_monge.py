import numpy as np

from universal_autoencoder.monge import fit_pca_monge_chart


def test_pca_monge_reconstructs_tiny_synthetic_surface():
    x = np.linspace(-0.2, 0.2, 7)
    y = np.linspace(-0.2, 0.2, 7)
    xx, yy = np.meshgrid(x, y)
    zz = 0.1 * xx**2 - 0.05 * yy**2
    points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
    chart = fit_pca_monge_chart(points)
    assert chart.reconstruction_mse(points) < 1e-8
