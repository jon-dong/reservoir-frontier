import numpy as np

from .utils import linear_regression


def extract_edges(X):
    """
    Detects edges in a signed 2D array by identifying sign changes.

    For each 2x2 cell in the array, an edge is detected if the four corner
    values contain both positive and negative signs.

    Args:
        X: 2D numpy array with signed values

    Returns:
        Binary numpy array of shape (H-1, W-1) where True indicates an edge
    """

    Y = np.stack((X[1:, 1:], X[:-1, 1:], X[1:, :-1], X[:-1, :-1]), axis=-1)
    Z = np.max(Y, axis=-1) * np.min(Y, axis=-1)
    return Z < 0


def count_boxes(field: np.ndarray, box_size: int):
    """
    Counts boxes of a given size that contain at least one True value.

    Used in box-counting method for fractal dimension estimation. Divides the
    2D boolean array into a grid of box_size×box_size boxes and counts how many
    boxes contain at least one True value.

    Args:
        field: 2D boolean numpy array
        box_size: Size of each box (must be >= 1)

    Returns:
        Number of occupied boxes
    """
    if box_size < 1:
        raise ValueError("box_size must be at least 1")
    H, W = field.shape

    # Pad array to make dimensions divisible by box_size
    pad_H = (box_size - H % box_size) % box_size
    pad_W = (box_size - W % box_size) % box_size

    if pad_H > 0 or pad_W > 0:
        padded = np.pad(field, ((0, pad_H), (0, pad_W)), mode='constant', constant_values=False)
    else:
        padded = field

    H_padded, W_padded = padded.shape
    n_rows = H_padded // box_size
    n_cols = W_padded // box_size

    # Reshape into blocks: (n_rows, box_size, n_cols, box_size)
    reshaped = padded.reshape(n_rows, box_size, n_cols, box_size)

    # Check if any element in each block is True (reduce over box dimensions)
    occupied = np.any(reshaped, axis=(1, 3))

    # Count occupied boxes
    N_boxes = np.sum(occupied)

    return N_boxes


def compute_dim(X, min_idx, max_idx, scales=list(range(1, 50))):
    """
    Computes fractal dimension using the box-counting method.

    Counts occupied boxes at multiple scales, then estimates the fractal
    dimension as the negative slope of the log-log plot of box count vs box size.
    Linear regression is performed on a subset of scales specified by min_idx and max_idx.

    Args:
        X: 2D boolean numpy array (typically output from extract_edges)
        min_idx: Start index for linear regression range
        max_idx: End index for linear regression range
        scales: List of box sizes to test (default: range(1, 50))

    Returns:
        Tuple of (fractal_dim, intercept, log_count, log_scales) where:
        - fractal_dim: Estimated fractal dimension (-slope)
        - intercept: Y-intercept of the regression line
        - log_count: Log of box counts at all scales
        - log_scales: Log of all box sizes
    """
    count_through_scales = []
    for scale in scales:
        count = count_boxes(X, scale)
        count_through_scales.append(count)

    log_count = np.log(np.array(count_through_scales))
    log_scales = np.log(np.array(scales))
    H, V = linear_regression(log_scales[min_idx:max_idx], log_count[min_idx:max_idx])

    return -H, V, log_count, log_scales
