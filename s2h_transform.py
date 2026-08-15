import numpy as np
from scipy.special import lpmv
import cv2
import math

def get_brain_mask(gray_image, n_phi_boundary=360):
    img = gray_image.astype(np.uint8) if gray_image.dtype != np.uint8 else gray_image
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        h, w = img.shape
        cx, cy = w / 2, h / 2
        R = np.full(n_phi_boundary, min(h, w) / 2)
        return mask, (cx, cy), R

    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        h, w = img.shape
        cx, cy = w / 2, h / 2
    else:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

    pts = largest.reshape(-1, 2).astype(np.float64)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    phis = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    radii = np.sqrt(dx ** 2 + dy ** 2)

    phi_bins = np.linspace(0, 2 * np.pi, n_phi_boundary, endpoint=False)
    R_of_phi = np.zeros(n_phi_boundary)
    bin_width = 2 * np.pi / n_phi_boundary
    for i, phi_b in enumerate(phi_bins):
        sel = np.abs(np.mod(phis - phi_b + np.pi, 2 * np.pi) - np.pi) < bin_width
        R_of_phi[i] = radii[sel].max() if sel.any() else np.median(radii)

    kernel = np.ones(5) / 5
    R_of_phi = np.convolve(np.concatenate([R_of_phi[-2:], R_of_phi, R_of_phi[:2]]),
                            kernel, mode="valid")

    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)

    return clean_mask, (cx, cy), R_of_phi


def _interp_R(phi, R_of_phi):
    n = len(R_of_phi)
    phi = np.mod(phi, 2 * np.pi)
    idx = phi / (2 * np.pi) * n
    i0 = np.floor(idx).astype(int) % n
    i1 = (i0 + 1) % n
    frac = idx - np.floor(idx)
    return (1 - frac) * R_of_phi[i0] + frac * R_of_phi[i1]

def hemispherical_embed(gray_image, n_theta=32, n_phi=64,
                         theta_max=np.pi / 2, mask=None, centroid=None, R_of_phi=None):
    img = gray_image.astype(np.float32)
    if mask is None or centroid is None or R_of_phi is None:
        mask, centroid, R_of_phi = get_brain_mask(gray_image)
    cx, cy = centroid

    theta_grid = np.linspace(0, theta_max, n_theta)
    phi_grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    theta_mesh, phi_mesh = np.meshgrid(theta_grid, phi_grid, indexing="ij")
    # Exact inverse of theta = arccos(1 - rho^2): rho = sqrt(1 - cos(theta))
    rho_mesh = np.sqrt(np.clip(1 - np.cos(theta_mesh), 0, 1))
    R_mesh = _interp_R(phi_mesh, R_of_phi)
    r_mesh = rho_mesh * R_mesh

    x_mesh = cx + r_mesh * np.cos(phi_mesh)
    y_mesh = cy + r_mesh * np.sin(phi_mesh)

    h, w = img.shape
    x_mesh = np.clip(x_mesh, 0, w - 1)
    y_mesh = np.clip(y_mesh, 0, h - 1)

    I_grid = cv2.remap(
        img, x_mesh.astype(np.float32), y_mesh.astype(np.float32),
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

    return I_grid, theta_grid, phi_grid, centroid, R_of_phi


def inverse_map_point(theta0, phi0, centroid, R_of_phi):
    """Map a single (theta, phi) hemisphere point back to pixel (x, y)."""
    cx, cy = centroid
    rho = np.sqrt(np.clip(1 - np.cos(theta0), 0, 1))
    R = _interp_R(np.array([phi0]), R_of_phi)[0]
    r = rho * R
    x = cx + r * np.cos(phi0)
    y = cy + r * np.sin(phi0)
    return x, y


_BASIS_CACHE = {}


def build_sector_harmonic_basis(theta_grid, phi_grid, max_degree=8,
                                 phi_sector=None, theta_sector=None):
    key = (tuple(theta_grid), tuple(phi_grid), max_degree,
           phi_sector, theta_sector)
    if key in _BASIS_CACHE:
        return _BASIS_CACHE[key]

    n_theta, n_phi = len(theta_grid), len(phi_grid)
    theta_mesh, phi_mesh = np.meshgrid(theta_grid, phi_grid, indexing="ij")

    in_sector = np.ones_like(theta_mesh, dtype=bool)
    if theta_sector is not None:
        in_sector &= (theta_mesh >= theta_sector[0]) & (theta_mesh <= theta_sector[1])
    if phi_sector is not None:
        lo, hi = phi_sector
        phi_wrapped = np.mod(phi_mesh - lo, 2 * np.pi)
        in_sector &= phi_wrapped <= np.mod(hi - lo, 2 * np.pi)

    cos_theta = np.cos(theta_mesh)
    weights_2d = np.sin(theta_mesh)  # spherical area element
    weights_2d = np.where(in_sector, weights_2d, 0.0)
    weights = weights_2d.flatten()

    candidates = []
    for n in range(max_degree + 1):
        for m in range(-n, n + 1):
            P = lpmv(abs(m), n, cos_theta)
            P = np.nan_to_num(P)
            if m < 0:
                ang = np.sin(abs(m) * phi_mesh)
            elif m == 0:
                ang = np.ones_like(phi_mesh)
            else:
                ang = np.cos(m * phi_mesh)
            f = P * ang
            f = np.where(in_sector, f, 0.0)
            candidates.append(f.flatten())

    B = np.stack(candidates, axis=1)  # (n_theta*n_phi, num_candidates)

    # Weighted Gram-Schmidt via QR on sqrt(weight)-scaled columns
    sqrt_w = np.sqrt(np.maximum(weights, 0))
    B_weighted = B * sqrt_w[:, None]

    # drop near-zero columns (can happen for high m outside a narrow sector)
    col_norms = np.linalg.norm(B_weighted, axis=0)
    keep = col_norms > 1e-8
    B_weighted = B_weighted[:, keep]

    Q, _ = np.linalg.qr(B_weighted)
    # undo the weighting so basis is expressed back in the original (unweighted) space
    safe_sqrt_w = np.where(sqrt_w > 1e-8, sqrt_w, 1.0)
    basis = Q / safe_sqrt_w[:, None]

    _BASIS_CACHE[key] = (basis, weights)
    return basis, weights

def compute_q1_q2(theta1, theta2, a=-1.0, b=1.0):

    denom = np.cos(theta1) - np.cos(theta2)
    if abs(denom) < 1e-12:
        raise ValueError("theta1 and theta2 must differ (degenerate sector).")
    q1 = (b - a) / denom
    q2 = a - q1 * np.cos(theta2)
    return q1, q2


def shifted_alp(n, m, x, q1, q2):

   # P~_n^m(x) = P_n^m(q1*x + q2)

    arg = q1 * x + q2
    arg = np.clip(arg, -1.0, 1.0)  # guard tiny float overshoot at sector edges
    return lpmv(m, n, arg)


_EXACT_BASIS_CACHE = {}


def build_s2h_basis_exact(theta_grid, phi_grid, max_degree,
                           theta_sector, phi_sector):
    key = (tuple(theta_grid), tuple(phi_grid), max_degree, theta_sector, phi_sector)
    if key in _EXACT_BASIS_CACHE:
        return _EXACT_BASIS_CACHE[key]

    theta1, theta2 = theta_sector
    phi1, phi2 = phi_sector
    q1, q2 = compute_q1_q2(theta1, theta2)
    u = 2 * np.pi / (phi2 - phi1)

    theta_mesh, phi_mesh = np.meshgrid(theta_grid, phi_grid, indexing="ij")
    in_sector = (
        (theta_mesh >= theta1) & (theta_mesh <= theta2) &
        (phi_mesh >= phi1) & (phi_mesh <= phi2)
    )
    cos_theta = np.cos(theta_mesh)
    weights_2d = np.where(in_sector, np.sin(theta_mesh), 0.0)
    weights = weights_2d.flatten()

    columns = []
    for n in range(max_degree + 1):
        for m in range(0, n + 1):
            P_tilde = np.nan_to_num(shifted_alp(n, m, cos_theta, q1, q2))
            K = np.sqrt(
                (2 * n + 1) * math.factorial(n - m) * q1 * u
                / (4 * np.pi * math.factorial(n + m))
            )
            phase = m * u * (phi_mesh - phi1)  # phi shifted so sector starts at phase 0

            if m == 0:
                f = K * P_tilde  # real, e^0 = 1
                f = np.where(in_sector, f, 0.0)
                columns.append(f.flatten())
            else:
                f_cos = np.sqrt(2) * K * P_tilde * np.cos(phase)
                f_sin = np.sqrt(2) * K * P_tilde * np.sin(phase)
                f_cos = np.where(in_sector, f_cos, 0.0)
                f_sin = np.where(in_sector, f_sin, 0.0)
                columns.append(f_cos.flatten())
                columns.append(f_sin.flatten())

    basis = np.stack(columns, axis=1)
    result = (basis, weights, (q1, q2, u))
    _EXACT_BASIS_CACHE[key] = result
    return result


def s2h_transform(I_grid, basis, weights):
    """Forward transform: image (on hemisphere grid) -> S2H coefficients."""
    I_flat = I_grid.flatten()
    # projection: c_k = sum(I * basis_k * weight)
    coeffs = (I_flat * weights) @ basis
    return coeffs


def s2h_inverse(coeffs, basis, grid_shape):
    """Inverse (band-limited) reconstruction from S2H coefficients."""
    I_flat = basis @ coeffs
    return I_flat.reshape(grid_shape)


def extract_s2h_features(gray_image, n_theta=24, n_phi=48, max_degree=6,
                          theta_sector=None, phi_sector=(0.0, 2 * np.pi)):
    
    I_grid, theta_grid, phi_grid, centroid, R_of_phi = hemispherical_embed(
        gray_image, n_theta=n_theta, n_phi=n_phi
    )
    if theta_sector is None:
        theta_sector = (theta_grid[0], theta_grid[-1])

    basis, weights, (q1, q2, u) = build_s2h_basis_exact(
        theta_grid, phi_grid, max_degree, theta_sector, phi_sector
    )
    coeffs = s2h_transform(I_grid, basis, weights)
    return coeffs, (theta_grid, phi_grid, centroid, R_of_phi, basis, weights, q1, q2, u)


def project_grid_to_pixels(grid, theta_grid, phi_grid, centroid, R_of_phi, image_shape):
    
    h, w = image_shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = centroid
    dx = (xx - cx).astype(np.float64)
    dy = (yy - cy).astype(np.float64)
    r = np.sqrt(dx ** 2 + dy ** 2)
    phi = np.mod(np.arctan2(dy, dx), 2 * np.pi)

    R = _interp_R(phi, R_of_phi)
    rho = np.clip(r / np.maximum(R, 1e-6), 0, 1)
    theta = np.arccos(np.clip(1 - rho ** 2, -1, 1))

    n_theta, n_phi = len(theta_grid), len(phi_grid)
    theta_span = theta_grid[-1] - theta_grid[0] if theta_grid[-1] != theta_grid[0] else 1.0
    ti = np.clip(((theta - theta_grid[0]) / theta_span * (n_theta - 1)).astype(int), 0, n_theta - 1)
    pj = np.mod((phi / (2 * np.pi) * n_phi).astype(int), n_phi)

    pixel_map = grid[ti, pj]
    # zero outside the brain boundary (rho was clipped to 1, so this masks
    # points that were actually outside R(phi))
    outside = r > R
    pixel_map = np.where(outside, 0.0, pixel_map)
    return pixel_map


def _shortest_interval_capturing_mass(grid, profile, mass_fraction, circular=False):
    
    n = len(grid)
    total = profile.sum()
    if total <= 0:
        return grid[0], grid[-1]
    target = mass_fraction * total

    if circular:
        # duplicate the profile to allow windows that wrap past the end
        ext_profile = np.concatenate([profile, profile])
        ext_grid = np.concatenate([grid, grid + (grid[-1] - grid[0] + (grid[1] - grid[0]))])
        cumsum = np.concatenate([[0], np.cumsum(ext_profile)])
        best_len = None
        best = (grid[0], grid[-1])
        for start in range(n):
            # binary-search-free linear scan (grids are small: <=few hundred)
            for end in range(start, start + n):
                mass = cumsum[end + 1] - cumsum[start]
                if mass >= target:
                    length = ext_grid[end] - ext_grid[start]
                    if best_len is None or length < best_len:
                        best_len = length
                        best = (ext_grid[start], ext_grid[end])
                    break
        return best
    else:
        cumsum = np.concatenate([[0], np.cumsum(profile)])
        best_len = None
        best = (grid[0], grid[-1])
        for start in range(n):
            for end in range(start, n):
                mass = cumsum[end + 1] - cumsum[start]
                if mass >= target:
                    length = grid[end] - grid[start]
                    if best_len is None or length < best_len:
                        best_len = length
                        best = (grid[start], grid[end])
                    break
        return best


def calibrate_sector_from_data(gray_images, n_theta=24, n_phi=48,
                                variance_capture=0.85, min_span_deg=20.0):
   
    grids = []
    theta_grid = phi_grid = None
    for img in gray_images:
        I_grid, theta_grid, phi_grid, _, _ = hemispherical_embed(img, n_theta=n_theta, n_phi=n_phi)
        grids.append(I_grid)
    stack = np.stack(grids, axis=0)  # (num_images, n_theta, n_phi)
    var_map = stack.var(axis=0)

    theta_profile = var_map.sum(axis=1)  # (n_theta,)
    phi_profile = var_map.sum(axis=0)    # (n_phi,)

    theta1, theta2 = _shortest_interval_capturing_mass(
        theta_grid, theta_profile, variance_capture, circular=False
    )
    phi1, phi2 = _shortest_interval_capturing_mass(
        phi_grid, phi_profile, variance_capture, circular=True
    )

    # guard against a degenerate/too-narrow sector (numerically unstable q1,q2)
    min_span = np.deg2rad(min_span_deg)
    if theta2 - theta1 < min_span:
        mid = (theta1 + theta2) / 2
        theta1, theta2 = max(theta_grid[0], mid - min_span / 2), min(theta_grid[-1], mid + min_span / 2)
    if np.mod(phi2 - phi1, 2 * np.pi) < min_span:
        mid = (phi1 + phi2) / 2
        phi1, phi2 = mid - min_span / 2, mid + min_span / 2

    return (float(theta1), float(theta2)), (float(phi1), float(phi2)), var_map


def s2h_degree_indices(max_degree):
   
    degrees = []
    for n in range(max_degree + 1):
        for m in range(0, n + 1):
            if m == 0:
                degrees.append(n)
            else:
                degrees.append(n)
                degrees.append(n)
    return np.array(degrees)
