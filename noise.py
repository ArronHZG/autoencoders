import numpy as np


def add_noise(img, noise_type="gaussian"):
    row, col = 28, 28
    img = img.astype(np.float32)

    if noise_type == "gaussian":
        mean = 0
        var = 10
        sigma = var ** .5
        noise = np.random.normal(-5.9, 5.9, img.shape)
        noise = noise.reshape(row, col)
        img = img + noise
        return img

    if noise_type == "speckle":
        noise = np.random.randn(row, col)
        noise = noise.reshape(row, col)
        img = img + img * noise
        return img
