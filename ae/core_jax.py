from collections.abc import Iterable
from flax import nnx
import jax.numpy as jnp
import jax
from jaxlib.mlir.dialects.sparse_tensor import out

class Encoder(nnx.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features: int | Iterable[int], *, rngs: nnx.Rngs):
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.hidden_features: list[int]
        if isinstance(hidden_features, int):
            self.hidden_features = [hidden_features]
        else:
            self.hidden_features = list(hidden_features)
        self.layers: list[nnx.Linear]
        for temp_in, temp_out in zip([self.in_features] + self.hidden_features, self.hidden_features + [self.out_features]):
            self.layers.append(nnx.Linear(in_features=temp_in, out_features=temp_out, rngs=rngs))

    def __call__(self, x: jax.Array, activation_fn = nnx.relu) -> jax.Array:
        h: jax.Array = x
        for layer in self.layers[:-1]:
            h = activation_fn(layer(h))
        h = self.layers[-1](h)
        return h

class Decoder(nnx.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features: int | Iterable[int], hidden_reversed: bool = True, *, rngs: nnx.Rngs):
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.hidden_features: list[int]
        if isinstance(hidden_features, int):
            self.hidden_features = [hidden_features]
        else:
            self.hidden_features = list(hidden_features)
        if hidden_reversed: self.hidden_features = self.hidden_features[::-1]
        self.layers: list[nnx.Linear]
        for temp_in, temp_out in zip([self.in_features] + self.hidden_features, self.hidden_features + [self.out_features]):
            self.layers.append(nnx.Linear(in_features=temp_in, out_features=temp_out, rngs=rngs))

    def __call__(self, x: jax.Array, activation_fn = nnx.relu) -> jax.Array:
        h: jax.Array = x
        for layer in self.layers[:-1]:
            h = activation_fn(layer(h))
        h = self.layers[-1](h)
        return h


