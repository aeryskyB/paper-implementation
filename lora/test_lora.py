import jax.numpy as jnp
from jax import random, Array, vmap, grad, jit, debug
from typing import List
from time import monotonic

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

def ffn_fwd(params, x):
    (fw, fb), params = params[-1], params[:-1]
    y = x
    for (w, b) in params:
        y = relu(w @ y + b)
    y = fw @ y + fb
    return y

batched_ffn_fwd = vmap(ffn_fwd, (None, 0), 0)

def ffn_fwd_with_lora(params, lora_params, x):
    (fw, fb), params = params[-1], params[:-1]
    (s, flb, fla), lora_params = lora_params[-1], lora_params[:-1]
    y = x
    for (w, b), (s, lb, la) in zip(params, lora_params):
        y = relu((w + s*lb@la) @ y + b)
    y = (fw + s*flb@fla) @ y + fb
    # debug.print("👀{x}", x=y.shape)
    return y

batched_ffn_fwd_with_lora = vmap(ffn_fwd_with_lora, (None, None, 0), 0)

def mse_loss(y_pred, y):
    return jnp.sum((y - y_pred)**2 / y.shape[0])

def predict(params, x, y):
    y_pred = batched_ffn_fwd(params, x)
    return mse_loss(y_pred, y)

def predict_lora(params, lora_params, x, y):
    y_pred = batched_ffn_fwd_with_lora(params, lora_params, x)
    # debug.print("👀{x}", x=y_pred.shape)
    return mse_loss(y_pred, y)

@jit
def update(params, x, y, lr):
    grads = grad(predict, 0)(params, x, y)
    return [(w - lr*dw, b - lr*db) for (w, b), (dw, db) in zip(params, grads)]

@jit
def update_lora(params, lora_params, x, y, lr):
    # debug.print("🥲{x}", x=lora_params[0])
    lora_grads = grad(predict_lora, 1)(params, lora_params, x, y)
    # debug.print("😭{x}", x=lora_grads[0])
    ## don't tune scaling factor
    updated_lora_params = [(s, b-lr*db, a-lr*da) for (s, b, a), (_, db, da) in zip(lora_params, lora_grads)]
    # debug.print("zzzzzzzz {x}", x=updated_lora_params[0][1].shape)
    # debug.print("kkkkkkkk {x}", x=lora_params[0][1].shape)
    return updated_lora_params

if __name__ == "__main__":
    keys = random.split(random.key(7), 2)
    lr = 2e-3
    batch_size = 128
    num = 10_000
    epoch = 10

    x = random.normal(key=keys[0], shape=(num, 1024))
    y = random.uniform(key=keys[1], shape=(num, 10))

    train_split = 0.8
    train_num = int(num * train_split)

    x_train = x[:train_num]
    y_train = y[:train_num]
    x_test = x[train_num:]
    y_test = y[train_num:]

    in_dim, out_dim = (1024, 10)

    # rather some pretrained weights
    dims = [in_dim] + [512, 128] + [out_dim]
    ffn_params = ffn(dims, key=keys[1])

    lora_keys = random.split(keys[0], len(dims)-1)
    lora_r = 4
    lora_alpha = lora_r
    loras = [Linear(in_dim=i_d, out_dim=o_d, r=lora_r, alpha=lora_alpha, key=k) \
        for i_d, o_d, k in zip(dims[:-1], dims[1:], lora_keys)]

    lora_params = [(l.scaling_factor, l.B, l.A) for l in loras]


    print("<-train the base ffn->")
    t = 0
    for e in range(epoch):
        t0 = monotonic()
        for i in range(0, num, batch_size):
            ffn_params = update(ffn_params,
                                x_train[i:i+batch_size],
                                y_train[i:i+batch_size],
                                lr=lr)
        dt = monotonic() - t0
        t += dt
        train_loss = predict(ffn_params, x_train, y_train)
        test_loss = predict(ffn_params, x_test, y_test)
        print(f"epoch {e:>2} | {train_loss=:<20} | {test_loss=}")
    print(f"time taken: {t}\n")

    lora_lr = 8e-4
    print("<-train lora->")
    t = 0
    for e in range(epoch):
        t0 = monotonic()
        for i in range(0, train_num, batch_size):
            lora_params = update_lora(ffn_params, lora_params,
                                      x_train[i:i+batch_size],
                                      y_train[i:i+batch_size],
                                      lr=lora_lr)
        dt = monotonic() - t0
        t += dt
        train_loss = predict_lora(ffn_params, lora_params, x_train, y_train)
        test_loss = predict_lora(ffn_params, lora_params, x_test, y_test)
        print(f"epoch {e:>2} | {train_loss=:<20} | {test_loss=}")
    print(f"time taken: {t}")

