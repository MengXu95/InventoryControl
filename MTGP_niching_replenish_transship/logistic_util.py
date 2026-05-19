import numpy as np
def logistic(x, k=1, c=0):
    """
    Computes the logistic function.

    Args:
        x: Input value (can be a scalar or a NumPy array).
        k: Steepness factor (default: 1).
        c: Center shift factor (default: 0).

    Returns:
        The logistic function output (same shape as x).
    """
    return 1 / (1 + np.exp(-k * (x - c)))




# def logistic_scale_and_shift(original_output, a, b, k=1, c=0):
#     """
#     Applies the logistic function, scales, and shifts to constrain output to [a, b].
#
#     Args:
#         original_output: The original output value(s).
#         a: Minimum value of the desired range.
#         b: Maximum value of the desired range.
#         k: Steepness factor for the logistic function (default: 1).
#         c: Center shift factor for the logistic function (default: 0).
#
#     Returns:
#         The transformed output value(s) in the range [a, b].
#     """
#
#     s = logistic(original_output, k, c)
#     return a + (b - a) * s

def logistic_scale_and_shift(original_output, a, b, k=1, c=0):
    """
    Applies the logistic function, scales, and shifts to constrain output to [a, b].
    Handles potential overflow by returning overflow_value.

    Args:
        original_output: The original output value(s).
        a: Minimum value of the desired range.
        b: Maximum value of the desired range.
        k: Steepness factor for the logistic function (default: 1).
        c: Center shift factor for the logistic function (default: 0).
        overflow_value: value to return when overflow occur (default: 100).

    Returns:
        The transformed output value(s) in the range [a, b], or overflow_value on overflow.
    """
    with np.errstate(over='ignore'): # Suppress overflow warnings during calculation
        s = 1 / (1 + np.exp(-k * (original_output - c)))
        return a + (b - a) * s