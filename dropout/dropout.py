import jax
from typing import Union

def dropout(y: jax.Array, dropout_p: float = 0.5, key: Union[None, jax.Array] = None, is_training: bool = True) -> Union[jax.Array, AssertionError]:
    assert dropout_p >= 0 and dropout_p <= 1, "p must be in the range [0, 1]"

    retention_p = 1 - dropout_p

    if is_training:
        if key is None: key = jax.random.key(0)
        r = jax.random.bernoulli(key=key, p=retention_p, shape=y.shape)
        y_ = r * y
    else:
        y_ = retention_p * y

    return y_

if __name__ == "__main__":
    key = jax.random.key(2024)
    n = 5
    keys = jax.random.split(key, n+1)

    y = jax.random.normal(key=keys[-1], shape=(10,))
    print('y:\n', y)
    for i in range(n):
        y_ = dropout(y, dropout_p=0.1, key=keys[i])
        print('\ny_:\n', y_)

