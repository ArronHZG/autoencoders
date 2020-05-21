# AutoEncoder

Pytorch implementation of  various autoencoder.

Requirements:

pytorch >= 1.5.0

## AutoEncoder

$$
\mathcal{J}_{{AE}}(\theta)=\sum_{x \in D_{n}} L(x, g(f(x)))
$$

## L1 AutoEncoder

$$
\mathcal{J}_{{AE}+{L1}}(\theta)=\sum_{x \in D_{n}} L(x, g(f(x)))+\lambda \sum_{i j} |W_{i j}|
$$

## L2 AutoEncoder

$$
\mathcal{J}_{{AE}+{L2}}(\theta)=\sum_{x \in D_{n}} L(x, g(f(x)))+\lambda \sum_{i j} W_{i j}^{2}
$$

## Contractive AutoEncoder

$$
\mathcal{J}_{{CAE}}(\theta)=\sum_{x \in D_{n}}\left(L(x, g(f(x)))+\lambda\left\|J_{f}(x)\right\|_{F}^{2}\right)
$$

$$
\left\|J_{f}(x)\right\|_{F}^{2}=\sum_{i j}\left(\frac{\partial h_{j}(x)}{\partial x_{i}}\right)^{2}
$$

## Sparse AutoEncoder

[CS294a A ng](https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf)
$$
\mathcal{J}_{{SPARSE}}(\theta)=\sum_{x \in D_{n}} L(x, g(f(x)))+
\lambda \sum_{j=1}^{s_{2}} {KL}\left(\rho \| \hat{\rho}_{j}\right)
$$

## Denoising AutoEncoder

paper: "Extracting and Composing Robust Features"