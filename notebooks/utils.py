from typing import Tuple, Any, TypeVar, Callable

import numpy as np


F = TypeVar("F", bound=Callable[..., Any])


def x_sinx(x: np.ndarray) -> Any:
    """One-dimensional x*sin(x) function."""
    return x*np.sin(x)


def get_1d_data_with_constant_noise(
    func,
    minimum,
    maximum,
    n_samples,
    noise
):
    """
    Generate 1D noisy data uniformely from the given function
    and standard deviation for the noise.
    """
    X_train = np.linspace(minimum, maximum, n_samples)
    np.random.shuffle(X_train)
    X_test = np.linspace(minimum, maximum, n_samples*5)
    y_train, y_mesh, y_test = func(X_train), func(X_test), func(X_test)
    y_train += np.random.normal(0, noise, y_train.shape[0])
    y_test += np.random.normal(0, noise, y_test.shape[0])
    return X_train.reshape(-1, 1), y_train, X_test.reshape(-1, 1), y_test, y_mesh


def get_1d_data_with_normal_distrib(
    func: F,
    mu: float,
    sigma: float,
    n_samples: int,
    noise: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate noisy 1D data with normal distribution from given function
    and noise standard deviation.
    """
    np.random.seed(42)
    X_train = np.random.normal(mu, sigma, n_samples)
    X_test = np.arange(mu - 4*sigma, mu + 4*sigma, sigma/20.)
    y_train, y_mesh, y_test = func(X_train), func(X_test), func(X_test)
    y_train += np.random.normal(0, noise, y_train.shape[0])
    y_test += np.random.normal(0, noise, y_test.shape[0])
    return (
        X_train.reshape(-1, 1), y_train, X_test.reshape(-1, 1), y_test, y_mesh
    )


def plot_1d_data(
    X_train,
    y_train,
    X_test,
    y_test,
    y_sigma,
    y_pred,
    y_pred_low,
    y_pred_up,
    ax=None,
    title=None,
    set_limits=False
):
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.fill_between(X_test, y_pred_low, y_pred_up, alpha=0.3)
    ax.scatter(X_train, y_train, color="red", alpha=0.3, label="Training data")
    ax.plot(X_test, y_test, color="gray", label="True confidence intervals")
    ax.plot(X_test, y_test - y_sigma, color="gray", ls="--")
    ax.plot(X_test, y_test + y_sigma, color="gray", ls="--")
    ax.plot(X_test, y_pred, color="blue", alpha=0.5, label="Prediction intervals")
    if set_limits:
        ax.set_xlim([-10, 10])
        ax.set_ylim([np.min(y_test)*1.3, np.max(y_test)*1.3])
    if title is not None:
        ax.set_title(title)
    ax.legend()
