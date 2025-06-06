import torch.nn as nn


class FCNClassificationLoss(nn.Module):
    """
    Default FireCastNet classification loss.

    Parameters
    ----------
    pixel_weights : torch.Tensor
        Tensor of shape [1, H, W] defining the weight for each pixel.
    """

    def __init__(self, pixel_weights=None ):
        super().__init__()
        if pixel_weights is not None:
            self.register_buffer("pixel_weights", pixel_weights)
        else:
            self.pixel_weights = None
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, invar, outvar):
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

        final_loss = weighted_loss.mean()

        return final_loss
