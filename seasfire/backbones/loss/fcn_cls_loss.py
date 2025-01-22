import torch.nn as nn


class FCNClassificationLoss(nn.Module):
    """
    Default FireCastNet classification loss.

    Parameters
    ----------
    pixel_weights : torch.Tensor
        Tensor of shape [1, H, W] defining the weight for each pixel.
    """

    def __init__(self, pixel_weights=None, enable_clima=False, clima_lambda=0.1):
        super().__init__()
        if pixel_weights is not None:
            self.register_buffer("pixel_weights", pixel_weights)
        else:
            self.pixel_weights = None
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.enable_clima = enable_clima
        self.clima_lambda = clima_lambda

    def forward(self, invar, outvar, clima=None):
        """
        Forward function to compute weighted BCE loss.

        Parameters
        ----------
        invar : torch.Tensor
            Predictions of shape [B, H, W].
        outvar : torch.Tensor
            Target values of shape [B, H, W].
        """
        # Compute pixel-wise BCE loss
        pixel_loss = self.bce_loss(invar, outvar)

        if self.pixel_weights is not None:
            # Apply region weights to the loss
            weighted_loss = pixel_loss * self.pixel_weights
        else: 
            weighted_loss = pixel_loss

        if self.enable_clima and clima is not None:
            # Compute climatology deviation penalty (e.g., MSE between prediction and clima)
            clima_penalty = ((invar - clima) ** 2)
            # if self.pixel_weights is not None:
            #     clima_penalty = clima_penalty * self.pixel_weights

            # Also add the clima penalty
            clima_loss = self.clima_lambda * clima_penalty.mean()
        else: 
            clima_loss = 0

        # Reduce to the final loss value (mean over all dimensions)
        final_loss = weighted_loss.mean() + clima_loss

        return final_loss
