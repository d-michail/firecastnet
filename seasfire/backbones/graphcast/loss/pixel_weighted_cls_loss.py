import torch.nn as nn

class PixelWeightedBCEWithLogitsLoss(nn.Module):
    """
    BCEWithLogitsLoss with pixel-based weighting.

    Parameters
    ----------
    pixel_weights : torch.Tensor
        Tensor of shape [H, W] defining the weight for each pixel.
    """

    def __init__(self, pixel_weights):
        super().__init__()
        self.register_buffer("pixel_weights", pixel_weights)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')  # Compute per-pixel loss

    def forward(self, invar, outvar):
        """
        Forward function to compute weighted BCE loss.

        Parameters
        ----------
        invar : torch.Tensor
            Predictions of shape [C, H, W].
        outvar : torch.Tensor
            Target values of shape [C, H, W].
        """
        # Compute pixel-wise BCE loss
        pixel_loss = self.bce_loss(invar, outvar)

        # Apply region weights to the loss
        weighted_loss = pixel_loss * self.pixel_weights

        # Reduce to the final loss value (mean over all dimensions)
        final_loss = weighted_loss.mean()

        return final_loss

