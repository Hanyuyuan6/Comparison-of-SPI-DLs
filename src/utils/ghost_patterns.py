import scipy.linalg
import numpy as np

def get_hadamard_matrix(img_pixels, bucket_size):
    """
    Generate a submatrix of a Hadamard matrix for ghost imaging patterns.

    Args:
        img_pixels (int): Number of pixels in the flattened image (width * height).
        bucket_size (int): Number of bucket detectors (number of Hadamard rows required).

    Returns:
        numpy.ndarray: A (bucket_size, img_pixels) Hadamard submatrix.

    Raises:
        ValueError: If ``img_pixels`` is not a power of two. A Hadamard matrix exists
            only at power-of-two orders; column-truncating a padded one would return
            non-orthogonal rows (Phi @ Phi.T != N*I), an invalid measurement operator.
    """
    # Hadamard matrices exist only at power-of-two orders. img_pixels must be one
    # exactly -- otherwise the column slice below truncates a larger Hadamard and the
    # rows silently lose orthogonality (Phi @ Phi.T != N*I), a physically invalid SPI
    # operator. Guard it loudly rather than return a wrong matrix.
    n = 2 ** int(np.ceil(np.log2(img_pixels)))
    if n != img_pixels:
        raise ValueError(
            f"img_pixels={img_pixels} is not a power of two. Column-truncating a "
            f"{n}x{n} Hadamard matrix to {img_pixels} columns returns NON-orthogonal "
            f"rows (Phi @ Phi.T != N*I), an invalid single-pixel measurement operator. "
            f"Use a pixel count that is a power of two (e.g. 128*128 = 16384)."
        )
    H = scipy.linalg.hadamard(n)
    # Return the first bucket_size rows and first img_pixels columns
    return H[:bucket_size, :img_pixels]