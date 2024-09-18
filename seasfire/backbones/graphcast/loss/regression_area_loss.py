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

        loss = (invar - outvar) ** 2
        loss = loss.mean(dim=(0))
        loss = torch.mul(loss, self.area.to(invar))
        loss = loss.mean()
        return loss


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
