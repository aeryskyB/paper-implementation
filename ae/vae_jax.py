from collections.abc import Iterable
from flax import nnx
from jax import numpy as jnp, random, Array
from .core_jax import Decoder

class EncoderVAE(nnx.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features: int | Iterable[int], *, rngs: nnx.Rngs):
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.hidden_features: list[int]
        if isinstance(hidden_features, int):
            self.hidden_features = [hidden_features]
        else:
            self.hidden_features = list(hidden_features)
        self.layers: list[nnx.Linear]
        layers = [self.in_features] + self.hidden_features
        for temp_in, temp_out in zip(layers[:-1], layers[1:]):
            self.layers.append(nnx.Linear(in_features=temp_in, out_features=temp_out, rngs=rngs))
        self.out_linear_mu = nnx.Linear(in_features=layers[-1], out_features=self.out_features, rngs=rngs)
        self.out_linear_log_var = nnx.Linear(in_features=layers[-1], out_features=self.out_features, rngs=rngs)

    def __call__(self, x: Array, activation_fn = nnx.relu) -> tuple[Array, Array]:
        h: Array = x
        for layer in self.layers[:-1]:
            h = activation_fn(layer(h))
        mu = self.out_linear_mu(h)
        log_var = nnx.softplus(self.out_linear_log_var(h))
        return mu, log_var


class VAE(nnx.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features: int | Iterable[int], activation_fn = nnx.relu, *, rngs: nnx.Rngs):
        self.encoder: EncoderVAE = EncoderVAE(in_features=in_features, out_features=out_features, hidden_features=hidden_features, rngs=rngs)
        self.decoder: Decoder = Decoder(in_features=out_features, out_features=in_features, hidden_features=hidden_features, rngs=rngs)
        self.activation_fn = activation_fn
        self.rngs = rngs

    def _reparam_trick(self, mu: Array, log_var: Array):
        sigma: Array = jnp.sqrt(jnp.exp(log_var))
        e: Array = random.normal(key=self.rngs.params(), shape=sigma.shape)
        z: Array = mu + sigma*e
        return z

    def __call__(self, x: Array):
        mu, log_var = self.encoder(x, activation_fn=self.activation_fn)
        z: Array = self._reparam_trick(mu, log_var)
        x_ = self.decoder(z)
        return x_


