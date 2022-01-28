"""
Taken from:
    https://github.com/danielkelshaw/ConcreteDropout/blob/master/condrop/concrete_dropout.py
It has been modified to allow true dropout with the learned probabilities at test time following:
    https://github.com/tjvandal/discrete-continuous-bdl/blob/master/bdl.py
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConcreteDropout(nn.Module):

    """
    Concrete Dropout.

    Implementation of the Concrete Dropout module as described in the
    'Concrete Dropout' paper: https://arxiv.org/pdf/1705.07832
    """

    def __init__(self,
                 weight_regulariser: float,
                 dropout_regulariser: float,
                 init_min: float = 0.1,
                 init_max: float = 0.1) -> None:

        """
        Concrete Dropout.

        Parameters
        ----------
        weight_regulariser : float
            Weight regulariser term.
        dropout_regulariser : float
            Dropout regulariser term.
        init_min : float
            Initial min value.
        init_max : float
            Initial max value.
        """

        super().__init__()

        self.weight_regulariser = weight_regulariser
        self.dropout_regulariser = dropout_regulariser

        init_min = np.log(init_min) - np.log(1.0 - init_min)
        init_max = np.log(init_max) - np.log(1.0 - init_max)

        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        self.p = torch.sigmoid(self.p_logit)

        self.regularisation = 0.0

    def forward(self, x: Tensor, layer: nn.Module, test: bool = False) -> Tensor:

        """
        Calculates the forward pass.

        The regularisation term for the layer is calculated and assigned to a
        class attribute - this can later be accessed to evaluate the loss.

        Parameters
        ----------
        x : Tensor
            Input to the Concrete Dropout.
        layer : nn.Module
            Layer for which to calculate the Concrete Dropout.
        test : bool
            Indicates whether to use true dropout (test=True) or to approximate it (test=False).
            Setting this to True should only be done at test/validation time hence the name.

        Returns
        -------
        Tensor
            Output from the dropout layer.
        """

        output = layer(self._concrete_dropout(x, test))

        sum_of_squares = 0
        for param in layer.parameters():
            sum_of_squares += torch.sum(torch.pow(param, 2))

        weights_reg = self.weight_regulariser * sum_of_squares / (1.0 - self.p)

        dropout_reg = self.p * torch.log(self.p)
        dropout_reg += (1.0 - self.p) * torch.log(1.0 - self.p)
        n, c, seq_len, h, w = x.shape
        dropout_reg *= self.dropout_regulariser * 27 * c / n    # not sure what to scale by here

        self.regularisation = weights_reg + dropout_reg

        return output

    def _concrete_dropout(self, x: Tensor, test: bool = False) -> Tensor:

        """
        Computes the Concrete Dropout.

        Parameters
        ----------
        x : Tensor
            Input tensor to the Concrete Dropout layer.
        test : bool
            Indicates whether to use true dropout (test=True) or to approximate it (test=False).
            Setting this to True should only be done at test/validation time hence the name.

        Returns
        -------
        Tensor
            Outputs from Concrete Dropout.
        """

        eps = 1e-7
        tmp = 0.1

        self.p = torch.sigmoid(self.p_logit)
        if test:    # At test time we perform actual dropout rather than its concrete approximation
            x = F.dropout3d(x, self.p, training=True)
        else:
            u_noise = torch.rand_like(x)

            drop_prob = (torch.log(self.p + eps) -
                         torch.log(1 - self.p + eps) +
                         torch.log(u_noise + eps) -
                         torch.log(1 - u_noise + eps))

            drop_prob = torch.sigmoid(drop_prob / tmp)

            random_tensor = 1 - drop_prob
            retain_prob = 1 - self.p

            x = torch.mul(x, random_tensor) / retain_prob

        return x


def concrete_regulariser(model: nn.Module) -> nn.Module:

    """Adds ConcreteDropout regularisation functionality to a nn.Module.
    Parameters
    ----------
    model : nn.Module
        Model for which to calculate the ConcreteDropout regularisation.
    Returns
    -------
    model : nn.Module
        Model with additional functionality.
    """

    def regularisation(self) -> Tensor:

        """Calculates ConcreteDropout regularisation for each module.
        The total ConcreteDropout can be calculated by iterating through
        each module in the model and accumulating the regularisation for
        each compatible layer.
        Returns
        -------
        Tensor
            Total ConcreteDropout regularisation.
        """

        total_regularisation = 0
        for module in filter(lambda x: isinstance(x, ConcreteDropout), self.modules()):
            total_regularisation += module.regularisation

        return total_regularisation

    setattr(model, 'regularisation', regularisation)

    return model
