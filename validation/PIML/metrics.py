import numpy as np
from scipy.ndimage import gaussian_gradient_magnitude

# -----------------------------
# 1. RMSE FREE SPACE
# -----------------------------
def rmse_free(corrected, reference, mask):
    diff = (corrected - reference) ** 2
    diff_masked = diff * mask
    return float(np.sqrt(np.mean(diff_masked + 1e-12)))

# -----------------------------
# 2. SMOOTHNESS INDEX
# (approximate gradient magnitude)
# -----------------------------
def smoothness_index(field):
    grad_mag = gaussian_gradient_magnitude(field, sigma=1)
    return float(np.mean(grad_mag))

# -----------------------------
# 3. BUILDING VIOLATION RATE
# (prediction leakage on buildings)
# -----------------------------
def building_violation(corrected, mask):
    building_mask = 1 - mask
    violations = corrected * building_mask
    return float(np.mean(violations > 0.0001))

# -----------------------------
# 4. WIND ALIGNMENT SCORE
# (gradient direction consistency)
# -----------------------------
def wind_alignment_score(field, wind_deg):
    # Convert wind angle into vector
    theta = np.radians(270 - wind_deg)
    wx = np.cos(theta)
    wy = np.sin(theta)

    # Compute gradients
    gx = np.diff(field, axis=1, prepend=field[:, :1])
    gy = np.diff(field, axis=0, prepend=field[:1, :])

    # Normalize gradient
    norm = np.sqrt(gx**2 + gy**2 + 1e-12)
    gx /= norm
    gy /= norm

    # Cosine similarity between gradient and wind vector
    align = gx * wx + gy * wy
    return float(np.mean(align))

# -----------------------------
# 5. GEO-MASK VIOLATION
# -----------------------------
def geomask_violation(corrected, mask):
    return float(np.mean((corrected > 0) * (1 - mask)))
