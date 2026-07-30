import numpy as np

def get_hadamard_matrix(img_pixels, bucket_size):
    """
    Generate a submatrix of a Hadamard matrix for ghost imaging patterns.

    Args:
        img_pixels (int): Number of pixels in the flattened image (width * height).
        bucket_size (int): Number of bucket detectors (number of Hadamard rows required).

    Returns:
        numpy.ndarray: A float32 ``(bucket_size, img_pixels)`` Hadamard
            submatrix in SciPy/Sylvester natural order.

    Raises:
        ValueError: If ``img_pixels`` is not a power of two. A Hadamard matrix exists
            only at power-of-two orders; column-truncating a padded one would return
            non-orthogonal rows (Phi @ Phi.T != N*I), an invalid measurement operator.
    """
    if not isinstance(img_pixels, (int, np.integer)) or img_pixels <= 0:
        raise ValueError(f"img_pixels must be a positive integer, got {img_pixels!r}")
    if not isinstance(bucket_size, (int, np.integer)) or not 1 <= bucket_size <= img_pixels:
        raise ValueError(
            f"bucket_size must satisfy 1 <= bucket_size <= img_pixels; got "
            f"bucket_size={bucket_size!r}, img_pixels={img_pixels!r}"
        )

    # Hadamard matrices exist only at power-of-two orders. img_pixels must be one
    # exactly -- otherwise the rows silently lose orthogonality.
    n = 2 ** int(np.ceil(np.log2(img_pixels)))
    if n != img_pixels:
        raise ValueError(
            f"img_pixels={img_pixels} is not a power of two. Column-truncating a "
            f"{n}x{n} Hadamard matrix to {img_pixels} columns returns NON-orthogonal "
            f"rows (Phi @ Phi.T != N*I), an invalid single-pixel measurement operator. "
            f"Use a pixel count that is a power of two (e.g. 128*128 = 16384)."
        )

    # A Sylvester Hadamard entry is (-1)^popcount(row & column).  Construct only
    # the requested rows instead of materializing the full N x N matrix (which is
    # 1 GiB at N=16384 even before conversion/copies).  Row-wise construction
    # keeps temporary memory O(N); the returned operator is O(MN), as required.
    columns = np.arange(img_pixels, dtype=np.uint32)
    patterns = np.empty((bucket_size, img_pixels), dtype=np.float32)
    for row in range(bucket_size):
        masked = np.bitwise_and(columns, np.uint32(row))
        masked ^= masked >> np.uint32(16)
        masked ^= masked >> np.uint32(8)
        masked ^= masked >> np.uint32(4)
        parity = (np.right_shift(np.uint32(0x6996), masked & np.uint32(0xF)) & 1)
        patterns[row] = 1.0 - 2.0 * parity.astype(np.float32)
    return patterns
