# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

class CellAreaWeightedMSELossFunction(nn.Module):
    def __init__(self):
        super(CellAreaWeightedMSELossFunction, self).__init__()

    def forward(self, predictions, targets, weights = None, mask=None):
        if mask is not None:
            predictions = predictions * mask
            targets = targets * mask
        loss = (predictions - targets) ** 2
        if weights is not None:
            if len(loss.shape) - len(weights.shape) == 1:
                weights = weights.unsqueeze(-1)
            elif abs(len(loss.shape) - len(weights.shape)) > 1:
                raise ValueError("weights must have the same number of dimensions as loss or one less to be broadcastable")
            loss *= weights
        return loss.mean()

# class CellAreaWeightedMSELossFunction(nn.Module):
#     """Loss function with cell area weighting.
# 
#     Parameters
#     ----------
#     area : torch.Tensor
#         Cell area with shape [H, W].
#     """
# 
#     def __init__(self, area):
#         super().__init__()
#         self.area = area
# 
#     def forward(self, invar, outvar):
#         """
#         Implicit forward function which computes the loss given
#         a prediction and the corresponding targets.
# 
#         Parameters
#         ----------
#         invar : torch.Tensor
#             prediction of shape [C, H, W].
#         outvar : torch.Tensor
#             target values of shape [C, H, W].
#         """
# 
#         loss = (invar - outvar) ** 2
#         loss = loss.mean(dim=(0))
#         loss = torch.mul(loss, self.area.to(invar))
#         loss = loss.mean()
#         return loss


class CellAreaWeightedL1LossFunction(nn.Module):
    """Loss function with cell area weighting.

    Parameters
    ----------
    area : torch.Tensor
        Cell area with shape [H, W].
    """

    def __init__(self, area):
        super().__init__()
        self.area = area

    def forward(self, invar, outvar):
        """
        Implicit forward function which computes the loss given
        a prediction and the corresponding targets.

        Parameters
        ----------
        invar : torch.Tensor
            prediction of shape [C, H, W].
        outvar : torch.Tensor
            target values of shape [C, H, W].
        """

        loss = torch.abs(invar - outvar)
        loss = loss.mean(dim=(0))
        loss = torch.mul(loss, self.area.to(invar))
        loss = loss.mean()
        return loss


class CellAreaWeightedHuberLossFunction(nn.Module):
    """Loss function with cell area weighting and Huber loss.

    Parameters
    ----------
    area : torch.Tensor
        Cell area with shape [H, W].
    delta : float
        Threshold parameter for Huber loss.
    """

    def __init__(self, area, delta=1.35):
        super().__init__()
        self.area = area
        self.delta = delta

    def forward(self, invar, outvar):
        """
        Implicit forward function which computes the loss given
        a prediction and the corresponding targets.

        Parameters
        ----------
        invar : torch.Tensor
            prediction of shape [C, H, W].
        outvar : torch.Tensor
            target values of shape [C, H, W].
        """

        # Compute the absolute difference
        diff = invar - outvar
        abs_diff = torch.abs(diff)

        # Compute the Huber loss
        quadratic = torch.where(abs_diff <= self.delta, 0.5 * diff ** 2, torch.zeros_like(diff))
        linear = torch.where(abs_diff > self.delta, self.delta * (abs_diff - 0.5 * self.delta), torch.zeros_like(diff))
        huber_loss = quadratic + linear

        # Mean over channels
        huber_loss = huber_loss.mean(dim=(0))

        # Apply area weighting
        weighted_loss = torch.mul(huber_loss, self.area.to(invar))

        # Mean over the spatial dimensions
        final_loss = weighted_loss.mean()

        return final_loss

