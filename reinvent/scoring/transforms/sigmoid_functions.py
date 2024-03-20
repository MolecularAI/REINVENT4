import numpy as np


def hard_sigmoid(x: np.ndarray, k: float) -> np.ndarray:
    return (k * x > 0).astype(np.float32)


def stable_sigmoid(x: np.ndarray, k: float, base_10: bool = True) -> np.ndarray:
    h = k * x
    if base_10:
        h = h * np.log(10)
    hp_idx = h >= 0
    y = np.zeros_like(x)
    y[hp_idx] = 1.0 / (1.0 + np.exp(-h[hp_idx]))
    y[~hp_idx] = np.exp(h[~hp_idx]) / (1.0 + np.exp(h[~hp_idx]))
    return y.astype(np.float32)


def double_sigmoid(
    x: np.ndarray,
    x_left: float,
    x_right: float,
    k: float,
    k_left: float,
    k_right: float,
) -> np.ndarray:
    """Compute double sigmoid based on stable sigmoid
    x: float or np.array
    x_left: float left sigmoid x value for which the output is 0.5 (low in previous implementation)
    x_right: float right sigmoid x value for which the output is 0.5 (high in previous implementation)
    k: float common scaling factor (coef_div in previous implementation)
    k_left: float scaling left factor (coef_si in previous implementation)
    k_right: float scaling right factor (coef_se in previous implementation)
    """
    x_center = (x_right - x_left) / 2 + x_left

    xl = x[x < x_center] - x_left
    xr = x[x >= x_center] - x_right

    if k == 0:
        sigmoid_left = hard_sigmoid(xl, k_left)
        sigmoid_right = 1 - hard_sigmoid(xr, k_right)
    else:
        k_left = k_left / k  # coef_si / coef_div
        k_right = k_right / k  # coef_se / coef_div
        sigmoid_left = stable_sigmoid(xl, k_left)
        sigmoid_right = 1 - stable_sigmoid(xr, k_right)

    d_sigmoid = np.zeros_like(x)
    d_sigmoid[x < x_center] = sigmoid_left
    d_sigmoid[x >= x_center] = sigmoid_right
    return d_sigmoid
