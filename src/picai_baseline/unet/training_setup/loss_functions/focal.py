#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss function for binary segmentation."""

    def __init__(self, alpha=1, gamma=2, num_classes=2, reduction="sum"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        targets = F.one_hot(targets, num_classes=self.num_classes).float()
        # targets = torch.moveaxis(targets, (0, 1, 2, 3, 4), (0, 2, 3, 4, 1))
        targets = torch.moveaxis(targets, (0, 1, 2, 3, 4,5), (0, 5, 2, 3, 4,1))[:, :, :, :, :, 0]


        #print(f"in forward inputs{inputs.shape}  targets {targets.shape}  ")
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        print(f" in loss ce_loss {type(ce_loss)}")

        # p_t = (inputs[-1] * targets[-1]) + ((1 - inputs[-1]) * (1 - targets[-1]))
        p_t = torch.add( torch.mul(inputs[-1] , targets[-1])  , torch.mul((1 - inputs[-1]) , (1 - targets[-1])))
        print(f" in loss p_t {type(p_t)}")

        loss = ce_loss * ((1 - p_t) ** self.gamma)
        print(f" in loss aaa {type(loss)}")
        if self.alpha >= 0:
            alpha_t = self.alpha * targets[-1] + (1 - self.alpha) * (1 - targets[-1])
            loss = torch.mul(loss, alpha_t)

            #loss = alpha_t * loss
            print(f" in loss bbb {type(loss)}")

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        print(f" in loss ccc {type(loss)}")

        return loss
