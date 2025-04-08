import jax, jax.numpy as jnp
from flax import nnx

class DownsampleBlock(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, rngs: nnx.Rngs):
        self.in_c = in_channels
        self.out_c = out_channels
        self.conv1 = nnx.Conv(self.in_c, self.out_c, kernel_size=(3, 3), padding="VALID", rngs=rngs)
        self.conv2 = nnx.Conv(self.out_c, self.out_c, kernel_size=(3, 3), padding="VALID", rngs=rngs)

    @nnx.jit
    def __call__(self, x):
        # conv1+relu -> conv2+relu -> max_pool
        x = nnx.relu(self.conv1(x))
        x_ = nnx.relu(self.conv2(x))
        x = nnx.max_pool(x_, window_shape=(2, 2), strides=(2, 2))
        return x, x_

class UpsampleBlock(nnx.Module):
    def __init__(self, in_channels: int, rngs: nnx.Rngs):
        self.in_c = in_channels
        self.convTr1 = nnx.ConvTranspose(self.in_c, self.in_c//2, kernel_size=(2, 2), strides=(2, 2), padding="VALID", rngs=rngs)
        self.conv1 = nnx.Conv(self.in_c, self.in_c//2, kernel_size=(3, 3), padding="VALID", rngs=rngs)
        self.conv2 = nnx.Conv(self.in_c//2, self.in_c//2, kernel_size=(3, 3), padding="VALID", rngs=rngs)

    @nnx.jit
    def __call__(self, x, x_):
        # convTr1 -> (concat) -> conv1+rele -> conv2+relu
        x = self.convTr1(x)
        crop_r = (x_.shape[0] - x.shape[0]) // 2
        crop_c = (x_.shape[1] - x.shape[1]) // 2
        x_crop = x_[crop_r:-crop_r, crop_c:-crop_c, :].copy()
        x = jnp.concat([x_crop, x], axis=-1)
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        return x

class UNet(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, layers=[64, 128, 256, 512], verbose=False):
        self.layers = [1] + layers
        # left/contraction
        self.left_stack = [DownsampleBlock(c1, c2, rngs) for (c1, c2) in zip(self.layers[:-1], self.layers[1:])]
        # middle
        self.mid_c = self.layers[-1]*2
        self.mid_conv1 = nnx.Conv(layers[-1], self.mid_c, kernel_size=(3, 3), padding="VALID", rngs=rngs)
        self.mid_conv2 = nnx.Conv(self.mid_c, self.mid_c, kernel_size=(3, 3), padding="VALID", rngs=rngs)
        # right/expansion
        self.layers = (self.layers + [self.mid_c])[::-1]
        self.layers[-1] = 2
        self.right_stack = [UpsampleBlock(c, rngs) for c in self.layers[:-2]]
        # last conv
        self.final_conv = nnx.Conv(self.layers[-2], self.layers[-1], kernel_size=(1, 1), padding="VALID", rngs=rngs)
        self.v = verbose

    def __call__(self, x):
        # (downsample)*n -> mid_conv1+relu -> mid_conv2+relu -> (dpsample)*n -> final_conv
        t = x
        left_traces = []
        for down in self.left_stack:
            t, t_ = down(t)
            left_traces.append(t_)
        t = nnx.relu(self.mid_conv1(t))
        t = nnx.relu(self.mid_conv2(t))
        for up in self.right_stack:
            t_ = left_traces.pop()
            t = up(t, t_)
        t = self.final_conv(t)
        return t

if __name__ == "__main__":
    # NOTE: nnx.Conv and nnx.ConvTranspose uses ...HWC format
    img_shape = (572, 572, 1)
    key = jax.random.key(0)
    x = jax.random.uniform(key, img_shape)

    rngs = nnx.Rngs(0)
    unet = UNet(rngs)

    y = unet(x)
    print(f"{x.shape=}")
    print(f"{y.shape=}")

