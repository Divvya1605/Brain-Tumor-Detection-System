"""
s2h_localization.py
====================
Localization and uncertainty estimation directly from the S2H domain
(no Grad-CAM needed for this part of the pipeline).

WHY NOT LITERAL MUSIC
-----------------------
The EEG paper's MUSIC-style spatial spectrum needs multiple time
snapshots to build a signal/noise covariance matrix. A single MRI slice
is one snapshot -- there is no time axis to exploit the same way. So
localization here uses a single-snapshot-appropriate substitute that
still lives natively in the S2H domain:

LOCALIZATION METHOD: per-class anomaly map via band-limited reconstruction
-----------------------------------------------------------------------------
1. During training, compute the MEAN S2H coefficient vector over all
   "notumor" training images: c_healthy_mean. This is the harmonic-domain
   "expected normal brain" profile.
2. For a new image, take its coefficients c and subtract the healthy mean:
   c_residual = c - c_healthy_mean
3. Reconstruct the RESIDUAL back onto the hemisphere grid using the same
   basis (this is just the inverse S2H transform of c_residual). Regions
   with large positive residual energy are regions that deviate from the
   "typical healthy brain" harmonic profile -- i.e. candidate lesion
   regions.
4. Map the peak residual region back through the inverse hemispherical
   map to pixel (x, y) coordinates for a bounding box, same as before.

This is a legitimate, defensible localization method that operates
natively in the harmonic domain (unlike Grad-CAM, which is a CNN gradient
method) -- but note explicitly in your paper that it is NOT the MUSIC
algorithm from the EEG reference; it's an anomaly-detection adaptation
suited to single-image (single-snapshot) data.

UNCERTAINTY METHOD
--------------------
Two complementary signals, combined:
  (a) Softmax entropy of the classifier output (same as your existing
      pipeline).
  (b) Truncation / residual energy ratio: how much energy in the S2H
      transform of the image lies OUTSIDE the low-order coefficients the
      classifier was trained on. An image whose energy is concentrated
      in the same low-order modes as training data is "in-distribution"
      (more trustworthy); an image with unusually high residual energy
      is atypical -> higher uncertainty. This mirrors the EEG paper's
      own use of eigenvalue/energy concentration as a confidence proxy.
"""

import numpy as np
import cv2

from s2h_transform import (hemispherical_embed, build_s2h_basis_exact,
                            s2h_transform, s2h_inverse, inverse_map_point,
                            get_brain_mask)


# ---------------------------------------------------------------------------
# 1. Healthy-mean coefficient profile (compute once over training data)
# ---------------------------------------------------------------------------

def compute_healthy_mean_coeffs(healthy_gray_images, n_theta=24, n_phi=48, max_degree=6,
                                 theta_sector=None, phi_sector=(0.0, 2 * np.pi)):
    """
    healthy_gray_images: list/iterable of 2D grayscale MRI slices labeled
                          'notumor'. Call this once over your training set
                          and save the result (e.g. np.save) -- no need to
                          recompute per inference.
    Returns: mean coefficient vector (num_coeffs,)
    """
    all_coeffs = []
    for img in healthy_gray_images:
        I_grid, theta_grid, phi_grid, centroid, R_of_phi = hemispherical_embed(
            img, n_theta=n_theta, n_phi=n_phi
        )
        sector = theta_sector if theta_sector is not None else (theta_grid[0], theta_grid[-1])
        basis, weights, _ = build_s2h_basis_exact(theta_grid, phi_grid, max_degree, sector, phi_sector)
        coeffs = s2h_transform(I_grid, basis, weights)
        all_coeffs.append(coeffs)
    return np.mean(all_coeffs, axis=0)


# ---------------------------------------------------------------------------
# 2. Localization: residual anomaly map -> bounding box
# ---------------------------------------------------------------------------

def localize_via_s2h_residual(gray_image, healthy_mean_coeffs,
                               n_theta=24, n_phi=48, max_degree=6,
                               threshold_ratio=0.5,
                               theta_sector=None, phi_sector=(0.0, 2 * np.pi)):
    """
    Returns: bbox (x, y, w, h) in ORIGINAL image pixel coordinates, or None,
             plus the residual anomaly map in hemisphere-grid space (for
             visualization) and the S2H coefficients (reusable for the
             classifier / uncertainty calc, avoids recomputation).
    """
    img_gray = gray_image if gray_image.ndim == 2 else cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)
    h, w = img_gray.shape

    I_grid, theta_grid, phi_grid, centroid, R_of_phi = hemispherical_embed(
        img_gray, n_theta=n_theta, n_phi=n_phi
    )
    sector = theta_sector if theta_sector is not None else (theta_grid[0], theta_grid[-1])
    basis, weights, _ = build_s2h_basis_exact(theta_grid, phi_grid, max_degree, sector, phi_sector)
    coeffs = s2h_transform(I_grid, basis, weights)

    residual_coeffs = coeffs - healthy_mean_coeffs
    residual_grid = s2h_inverse(residual_coeffs, basis, I_grid.shape)
    residual_energy = np.abs(residual_grid)

    # threshold the anomaly map
    if residual_energy.max() <= 1e-8:
        return None, residual_grid, coeffs

    thresh = residual_energy.max() * threshold_ratio
    mask = (residual_energy >= thresh).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None, residual_grid, coeffs

    # map each (theta, phi) grid cell with high residual back to pixel space
    px_coords = []
    for ti, pj in zip(ys, xs):
        theta0, phi0 = theta_grid[ti], phi_grid[pj]
        x, y = inverse_map_point(theta0, phi0, centroid, R_of_phi)
        px_coords.append((x, y))
    px_coords = np.array(px_coords)

    x_min, y_min = px_coords.min(axis=0)
    x_max, y_max = px_coords.max(axis=0)
    x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
    x_max, y_max = min(w - 1, int(x_max)), min(h - 1, int(y_max))

    bbox = (x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min))
    return bbox, residual_grid, coeffs


# ---------------------------------------------------------------------------
# 3. Uncertainty: entropy + S2H residual-energy ratio
# ---------------------------------------------------------------------------

def calculate_entropy_uncertainty(softmax_probs, num_classes):
    entropy = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10))
    max_entropy = np.log(num_classes)
    uncertainty = entropy / max_entropy
    return float(1 - uncertainty), float(uncertainty)


def calculate_s2h_novelty_score(coeffs, healthy_mean_coeffs, healthy_std_coeffs, scale=3.0):
    """
    Normalized deviation of this image's coefficients from the healthy
    training distribution, in units of standard deviation (z-score style).
    Returns a value in roughly [0, 1] after squashing, where higher means
    "further from what the model has seen" -> less trustworthy.

    healthy_std_coeffs: per-coefficient std dev over the healthy training
                         set (compute alongside compute_healthy_mean_coeffs).
    scale: CALIBRATED (not assumed) via calibrate_novelty_scale() -- the
           default of 3.0 was an arbitrary placeholder that saturates too
           easily, making even normal in-distribution images read as
           highly "novel" and capping certainty around 70-75% regardless
           of how confident/correct the prediction actually was.
    """
    z = (coeffs - healthy_mean_coeffs) / (healthy_std_coeffs + 1e-8)
    raw_score = np.sqrt(np.mean(z ** 2))
    novelty = 1 - np.exp(-raw_score / scale)  # squashes into [0,1), saturates gently
    return float(novelty)


def calibrate_novelty_scale(raw_scores, target_novelty_at_p90=0.3):
    """
    CALCULATES the novelty scale from data instead of assuming one. Run
    this once over held-out validation images (mix of all classes -- you
    want to know what "typical" looks like across the whole population,
    not just healthy ones) and reuse the result.

    Picks `scale` such that the 90th percentile of raw z-score energy
    (computed the same way as calculate_s2h_novelty_score) maps to
    `target_novelty_at_p90` -- i.e. even a somewhat atypical-but-normal
    image should only read as ~30% novel by default, leaving real
    headroom for genuinely unusual cases to score meaningfully higher.
    """
    raw_scores = np.asarray(raw_scores)
    p90 = np.percentile(raw_scores, 90)
    if p90 <= 1e-8:
        return 3.0  # degenerate fallback
    scale = -p90 / np.log(1 - target_novelty_at_p90)
    return float(scale)


def joint_uncertainty(softmax_probs, coeffs, healthy_mean_coeffs, healthy_std_coeffs,
                       num_classes, entropy_weight=0.6):
    """
    Combines classifier entropy with S2H-domain novelty into a single
    uncertainty score in [0, 1].
    """
    certainty, entropy_uncertainty = calculate_entropy_uncertainty(softmax_probs, num_classes)
    novelty = calculate_s2h_novelty_score(coeffs, healthy_mean_coeffs, healthy_std_coeffs)

    combined_uncertainty = entropy_weight * entropy_uncertainty + (1 - entropy_weight) * novelty
    combined_certainty = 1 - combined_uncertainty
    return combined_certainty, combined_uncertainty
