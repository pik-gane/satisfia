from satisfia.util.interval_tensor import IntervalTensor

from torch import Tensor, cat, stack, empty, zeros, ones, no_grad, minimum, maximum
from torch.nn import Module, Linear, ReLU, LayerNorm, Dropout, Parameter, ModuleList, ModuleDict
from more_itertools import pairwise
from math import sqrt
from typing import List, Dict, Callable

def concatenate_observations_and_aspirations( observation: Tensor,
                                              aspiration: IntervalTensor ) -> Tensor:
    
    return cat((observation, stack((aspiration.lower, aspiration.upper), dim=-1)), dim=-1)

class LinearReturningDict(Module):
    def __init__(self, in_features: int, out_features: Dict[str, int], bias: bool = True):
        super().__init__()
        self.linears = ModuleDict({ key: Linear(in_features, out_feats, bias=bias)
                                    for key, out_feats in out_features.items() })
        
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return {key: linear(x) for key, linear in self.linears.items()}
    
class NoisyLinear(Module):
    def __init__(self, in_features: int,
                       out_features: int,
                       same_noise_along_batch: bool = False,
                       batch_size: int | None = None,
                       bias: bool = True ):

        super().__init__()

        assert (batch_size is not None) == same_noise_along_batch
        
        self.in_features = in_features
        self.out_features = out_features
        self.same_noise_along_batch = same_noise_along_batch
        self.batch_size = batch_size

        self.weight = Parameter(empty(out_features, in_features))
        self.bias   = Parameter(empty(out_features)) if bias else None
        k = 1 / sqrt(self.in_features)
        self.weight.data.uniform_(-k, k)
        if bias:
            self.bias  .data.uniform_(-k, k)

        if self.same_noise_along_batch:
            self.weight_noise = Parameter(zeros(out_features, in_features))
            self.bias_noise   = Parameter(zeros(out_features)) if bias else None
        else:
            self.weight_noise = Parameter(zeros(batch_size, out_features, in_features))
            self.bias_noise   = Parameter(zeros(batch_size, out_features)) if bias else None
        
    def forward(self, x, noisy=True):
        assert x.dim() == 2
        if noisy and self.batch_size is not None:
            assert x.size(0) == self.batch_size

        weight = self.weight + self.weight_noise if noisy else self.weight
        bias   = self.bias   + self.bias_noise   if noisy else self.bias
        return (x.unsqueeze(-2) * weight).sum(-1) + bias

    @no_grad()
    def new_noise(self, std: float, which_in_batch: Tensor | None = None):
        k = 1 / sqrt(self.in_features)
        if self.same_noise_along_batch:
            self.weight_noise.data.normal_(k * std)
            if self.bias_noise is not None:
                self.bias_noise.data.normal_(k * std)
        else:
            if which_in_batch is None:
                which_in_batch = ones(self.batch_size, dtype=bool)
            self.weight_noise.data[which_in_batch, ...].normal_(k * std)
            if self.bias_noise is not None:
                self.bias_noise.data[which_in_batch, ...].normal_(k * std)
        
class NoisyLinearReturningDict(Module):
    def __init__( self, in_features: int,
                        out_features: Dict[str, int],
                        same_noise_along_batch: bool = False,
                        batch_size: int | None = None,
                        bias: bool = True ):
        
        super().__init__()
        self.noisy_linears = ModuleDict({
            key: NoisyLinear( in_features=in_features,
                              out_features=out_feats,
                              same_noise_along_batch=same_noise_along_batch,
                              batch_size=batch_size,
                              bias=bias )
            for key, out_feats in out_features.items()
        })

    def forward(self, x, noisy=True):
        return { key: noisy_linear(x, noisy=noisy)
                 for key, noisy_linear in self.noisy_linears.items() }

    @no_grad()
    def new_noise(self, std: float, which_in_batch: Tensor | None = None):
        for noisy_linear in self.noisy_linears.values():
            noisy_linear.new_noise(std=std, which_in_batch=which_in_batch)

class NoisyMLP(Module):
    def __init__(self, layer_sizes: List[int],
                       activation_function: Callable[[Tensor], Tensor] = ReLU(),
                       layer_norms: bool = True,
                       dropout: int | None = None,
                       same_noise_along_batch: bool = True,
                       batch_size: int | None = None,
                       final_activation_function: bool = False ):
        
        super().__init__()

        self.activation_function = activation_function
        self.final_activation_function = final_activation_function

        self.noisy_linears = ModuleList()
        for size_in, size_out in pairwise(layer_sizes):
            if isinstance(size_out, Dict):
                self.noisy_linears.append(
                    NoisyLinearReturningDict( size_in,
                                              size_out,
                                              same_noise_along_batch = same_noise_along_batch,
                                              batch_size = batch_size )
                )
            else:
                self.noisy_linears.append(
                    NoisyLinear( size_in,
                                 size_out,
                                 same_noise_along_batch = same_noise_along_batch,
                                 batch_size = batch_size )
                )

        if layer_norms:
            layer_norm_sizes = layer_sizes[1:] if final_activation_function else layer_sizes[1:-1]
            self.layer_norms = ModuleList(LayerNorm(size) for size in layer_norm_sizes)
        else:
            self.layer_norms = None

        if dropout is not None:
            self.dropout = Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x: Tensor, noisy: bool = True):
        for i, noisy_linear in enumerate(self.noisy_linears):
            x = noisy_linear(x, noisy=noisy)

            last_iteration = i == len(self.noisy_linears) - 1
            if not last_iteration or self.final_activation_function:
                if self.layer_norms is not None:
                    x = self.layer_norms[i](x)
        
                x = self.activation_function(x)
                
                if self.dropout is not None:
                    x = self.dropout(x)

        return x
    
    def new_noise(self, std: int, which_in_batch: Tensor | None = None):
        for noisy_linear in self.noisy_linears:
            noisy_linear.new_noise(std=std, which_in_batch=which_in_batch)


class MinMaxLayer(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, output: Dict[str, Tensor]) -> Dict[str, Tensor]:
        processed_output = {}
        for key, value in output.items():
            Qmin_k, Qmax_k = value[:, 0], value[:, 1]
            M_k = (Qmin_k + Qmax_k) / 2
            new_Qmin_k = minimum(Qmin_k, M_k)
            new_Qmax_k = maximum(Qmax_k, M_k)
            processed_output[key] = stack((new_Qmin_k, new_Qmax_k), dim=-1)
        return processed_output

class SatisfiaMLP(Module):
    def __init__(self, input_size: int,
                       output_not_depending_on_agent_parameters_sizes: Dict[str, int],
                       output_depending_on_agent_parameters_sizes: Dict[str, int],
                       common_hidden_layer_sizes: List[int],
                       hidden_layer_not_depending_on_agent_parameters_sizes: List[int],
                       hidden_layer_depending_on_agent_parameters_sizes: List[int],
                       same_noise_along_batch: bool = True,
                       batch_size: int | None = None,
                       activation_function: Callable[[Tensor], Tensor] = ReLU(),
                       dropout: int | None = 0.1,
                       layer_norms: bool = True ):

        super().__init__()

        self.common_layers = NoisyMLP(
            layer_sizes = [input_size] + common_hidden_layer_sizes,
            activation_function = activation_function,
            layer_norms = layer_norms,
            dropout = dropout,
            same_noise_along_batch = same_noise_along_batch,
            batch_size = batch_size,
            final_activation_function = True
        )

        last_common_layer_size = common_hidden_layer_sizes[-1]\
                                    if len(common_hidden_layer_sizes) > 0 \
                                        else input_size
        
        self.layers_not_depending_on_agent_parameters = NoisyMLP(
            layer_sizes =   [last_common_layer_size]
                          + hidden_layer_not_depending_on_agent_parameters_sizes
                          + [output_not_depending_on_agent_parameters_sizes],
            activation_function = activation_function,
            layer_norms = layer_norms,
            dropout = dropout,
            same_noise_along_batch = same_noise_along_batch,
            batch_size = batch_size
        )

        self.min_max_layer = MinMaxLayer()

        agent_parameters_size = 2

        self.layers_depending_on_agent_parameters = NoisyMLP(
            layer_sizes =   [last_common_layer_size + agent_parameters_size]
                          + hidden_layer_depending_on_agent_parameters_sizes
                          + [output_depending_on_agent_parameters_sizes],
            activation_function = activation_function,
            layer_norms = layer_norms,
            dropout = dropout,
            same_noise_along_batch = same_noise_along_batch,
            batch_size = batch_size
        )

    def forward(self, observations: Tensor, aspirations: Tensor, noisy: bool = True):
        agent_parameters_emebdding = stack((aspirations.lower, aspirations.upper), -1)

        common_hidden = self.common_layers(
            observations,
            noisy=noisy
        )

        output_not_depending_on_agent_parameters = self.layers_not_depending_on_agent_parameters(
            common_hidden,
            noisy = noisy
        )

        output_not_depending_on_agent_parameters = self.min_max_layer(
            output_not_depending_on_agent_parameters
        )

        output_depending_on_agent_parameters = self.layers_depending_on_agent_parameters(
            cat((common_hidden, agent_parameters_emebdding), -1),
            noisy = noisy
        )

        assert set(output_not_depending_on_agent_parameters.keys()) \
                   .isdisjoint(set(output_depending_on_agent_parameters.keys()))

        return output_not_depending_on_agent_parameters | output_depending_on_agent_parameters
    
    def new_noise(self, std: int, which_in_batch: Tensor | None = None):
        self.common_layers                           .new_noise(std, which_in_batch)
        self.layers_not_depending_on_agent_parameters.new_noise(std, which_in_batch)
        self.layers_depending_on_agent_parameters    .new_noise(std, which_in_batch)
