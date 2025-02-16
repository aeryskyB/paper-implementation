# based on: https://arxiv.org/pdf/2106.09685

import jax.numpy as jnp
from jax import random, Array


class Linear:
    def __init__(self, in_dim: int, out_dim: int, r: int, alpha: float, key: Array = random.key(0), dtype=jnp.float32):
        self.A = random.normal(key=key, shape=(r, in_dim), dtype=dtype)
        self.B = jnp.zeros(shape=(out_dim, r), dtype=dtype)
        self.r = r
        self.alpha = alpha
        self.scaling_factor = jnp.asarray(self.alpha / self.r, dtype=jnp.float32)

    def __call__(self, x: Array):
        # returns the partial product BAx during the forward pass
        return self.scaling_factor * self.B @ self.A @ x


