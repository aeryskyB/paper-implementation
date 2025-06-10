import jax.numpy as jnp
from jax import Array
from typing import List, Tuple

class Adam:
    def __init__(self, len_param_list: int, alpha: float = 1e-3, betas: Tuple[float, float] = (0.95, 0.95), eps: float =1e-8, dtype=jnp.float32):
        self.t = 1
        self.moments_1 = [jnp.zeros((1,), dtype=dtype)] * len_param_list
        self.moments_2 = [jnp.zeros((1,), dtype=dtype)] * len_param_list
        self.alpha = alpha
        self.beta_1 = betas[0]
        self.beta_2 = betas[1]
        self.eps = eps

    def __call__(self, params: List[Array], grads: List[Array]) -> List[Array]:
        """
        Returns updated params
        """
        updated_params = []
        for i, (param, gradient) in enumerate(zip(params, grads)):
            self.moments_1[i] = self.beta_1 * self.moments_1[i] + (1 - self.beta_1) * gradient
            self.moments_2[i] = self.beta_2 * self.moments_2[i] + (1 - self.beta_2) * gradient**2
            moment_1_ = self.moments_1[i] / (1 - self.beta_1**self.t)
            moment_2_ = self.moments_2[i] / (1 - self.beta_2**self.t)
            updated_params.append(param - self.alpha * moment_1_ / (jnp.sqrt(moment_2_) + self.eps))
        self.t = self.t + 1
        return updated_params

