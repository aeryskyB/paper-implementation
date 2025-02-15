import jax.numpy as jnp
from jax import random, Array, vmap, grad, jit
from typing import List

from lora import Linear

def linear_unit(in_dim, out_dim, key=random.key(0)):
    keys = random.split(key, 2)
    t = 1 / jnp.sqrt(in_dim)
    w = random.uniform(key=keys[0], shape=(out_dim, in_dim), minval=-t, maxval=t)
    b = random.uniform(key=keys[1], shape=(out_dim,), minval=-t, maxval=t)
    return w, b

def ffn(dims: List[int], key: Array = random.key(0)):
    n = len(dims) - 1
    keys = random.split(key, n)
    return [linear_unit(in_, out_, keys[i]) for i, (in_, out_) in enumerate(zip(dims[:-1], dims[1:]))]

def relu(x):
    return jnp.maximum(0, x)

def ffn_fwd_with_lora(params, lora_param, x):
    (lw, lb), params = params[-1], params[:-1]
    y = x
    for (w, b) in params:
        y = relu(w @ y + b)
    y = lw @ y + lb + lora_param @ x
    return y

batched_ffn_fwd_with_lora = vmap(ffn_fwd_with_lora, (None, None, 0), 0)

def mse_loss(y_pred, y):
    return jnp.sum((y - y_pred)**2 / y.shape[0])

def predict(params, lora_param, x, y):
    y_pred = batched_ffn_fwd_with_lora(params, lora_param, x)
    return mse_loss(y_pred, y)

@jit
def update_lora(params, lora_param, x, y, lr):
    lora_grad = grad(predict, 1)(params, lora_param, x, y)
    return lora_param - lr*lora_grad

if __name__ == "__main__":
    keys = random.split(random.key(7), 2)
    lr = 2e-4
    batch_size = 64
    num = 1_000
    epoch = 5

    x = random.normal(key=keys[0], shape=(num, 1024))
    y = random.uniform(key=keys[1], shape=(num, 10))

    train_split = 0.8
    train_num = int(num * train_split)

    x_train = x[:train_num]
    y_train = y[:train_num]
    x_test = x[train_num:]
    y_test = y[train_num:]

    in_dim, out_dim = (1024, 10)
    lora_test = Linear(in_dim=in_dim, out_dim=out_dim, r=4, key=keys[0])
    ba = lora_test()

    # rather some pretrained weights, 
    dims = [in_dim] + [512, 128] + [out_dim]
    ffn_params = ffn(dims, key=keys[1])

    for e in range(epoch):
        for i in range(0, num, batch_size):
            ba = update_lora(ffn_params, ba,
                             x_train[i*batch_size:(i+1)*batch_size],
                             y_train[i*batch_size:(i+1)*batch_size],
                             lr=lr)
        train_loss = predict(ffn_params, ba, x_train, y_train)
        test_loss = predict(ffn_params, ba, x_test, y_test)
        print(f"epoch {e} | {train_loss=} | {test_loss=}")

