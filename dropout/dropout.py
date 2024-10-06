import jax.numpy as jnp
import jax
from jax.typing import ArrayLike
from typing import Union

# during training time
def dropout(y: ArrayLike, p: float = 0.5, key: Union[None, jax.Array] = None, is_training: bool = True) -> jax.Array:
    assert p >= 0 and p <= 1, "p must be in the range [0, 1]"

    if is_training:
        if key is None: key = jax.random.key(0)
        r = jax.random.bernoulli(key=key, p=p, shape=y.shape)
        y_ = r * y
    else:
        y_ = p * y

    return y_

# during testing time:
#   y_ = p * y

if __name__ == "__main__":
    key = jax.random.key(2024)
    y = jax.random.normal(key=key, shape=(10,))
    y_ = dropout(y, key=key)
    print('y: ', y)
    print('y_: ', y_)
