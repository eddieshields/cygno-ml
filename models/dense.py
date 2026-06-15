"""Context-aware fully-connected network.

Provides a single ``Dense`` module that can optionally concatenate a global
context vector to its input at every forward call, enabling conditioning on
time embeddings, global image features, etc.
"""

from typing import Optional

import torch.nn as nn
from torch import Tensor

from models.utils import attach_context


class Dense(nn.Module):
    """Context-conditioned fully-connected feed-forward network.

    Builds an MLP from *input_size* to *output_size* with optional hidden
    layers, activation functions, layer normalisation, and dropout.

    When *context_size* > 0 the context vector is concatenated to the input
    before the first linear layer, allowing the network to condition its output
    on an external signal (e.g. a diffusion timestep embedding).

    Args:
        input_size:       Dimension of the primary input.
        output_size:      Dimension of the output.
        hidden_layers:    List of hidden-layer widths.  An empty list gives a
                          single linear layer.
        activation:       ``torch.nn`` activation class name for hidden layers.
                          Defaults to ``"ReLU"``.
        final_activation: ``torch.nn`` activation class name for the output
                          layer.  ``None`` returns raw logits.
        norm_layer:       ``torch.nn`` normalisation class name (e.g.
                          ``"LayerNorm"``).  Applied before each linear layer.
                          ``None`` disables normalisation.
        norm_final_layer: Whether to apply normalisation before the final
                          linear layer.  Defaults to ``False``.
        dropout:          Dropout probability applied before each linear layer.
                          ``0.0`` disables dropout.
        context_size:     Dimension of the optional context vector.  ``0``
                          disables context conditioning.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: list,
        activation: str = "ReLU",
        final_activation: Optional[str] = None,
        norm_layer: Optional[str] = None,
        norm_final_layer: bool = False,
        dropout: float = 0.0,
        context_size: int = 0,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size

        # First dimension is widened by context when conditioning is enabled
        node_list = [input_size + context_size, *hidden_layers, output_size]
        num_layers = len(node_list) - 1
        layers = []

        for i in range(num_layers):
            is_final = i == num_layers - 1
            apply_norm = norm_layer and (norm_final_layer or not is_final)

            if apply_norm:
                layers.append(getattr(nn, norm_layer)(node_list[i], elementwise_affine=False))
            if dropout and (norm_final_layer or not is_final):
                layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(node_list[i], node_list[i + 1]))

            if not is_final:
                layers.append(getattr(nn, activation)())
            elif final_activation:
                layers.append(getattr(nn, final_activation)())

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        """Forward pass.

        Args:
            x:       (..., input_size) input features.
            context: (..., context_size) conditioning vector.  Required when
                     ``context_size > 0``, otherwise ignored.

        Returns:
            (..., output_size) output tensor.
        """
        if self.context_size:
            x = attach_context(x, context)
        return self.net(x)
