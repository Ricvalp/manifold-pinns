import flax.linen as nn
from ml_collections import ConfigDict
from universal_autoencoder.upt_encoder import EncoderSupernodes
from universal_autoencoder.siren import ModulatedSIREN


class UniversalAutoencoder(nn.Module):
    """Universal Autoencoder model.

    This model combines a UPT encoder with a SIREN network for neural field generation.
    """

    cfg: ConfigDict

    def setup(self):

        # UPT encoder
        self.upt_encoder = EncoderSupernodes(
            cfg=self.cfg,
        )

        self.siren = ModulatedSIREN(
            cfg=self.cfg,
        )

    def __call__(self, points, supernode_idxs):
        """
        Args:
            points: Shape (batch_size, num_points, coord_dim)
            condition: Optional shape (batch_size, cond_dim)

        Returns:
            out: Output prediction for each point
        """
        # The encoder now returns both conditioning and transformed coordinates
        coords, conditioning = self.upt_encoder(points, supernode_idxs)

        # Use the transformed coordinates with the SIREN network
        out = self.siren(coords, conditioning)

        return out, coords, conditioning
