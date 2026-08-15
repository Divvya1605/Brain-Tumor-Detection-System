"""
s2h_inference.py
=================
Single-image inference pipeline for the Flask app, using the trained
hybrid CNN+S2H model plus the healthy-class S2H coefficient statistics
saved by train_s2h_ablation.py.

Produces, from one uploaded MRI:
  - classification (predicted class + confidence)
  - localization (bounding box, from the S2H residual anomaly map)
  - uncertainty (entropy + S2H novelty, combined)
  - a 3-panel image (original / heatmap overlay / bounding box) for the UI
"""

import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

from s2h_transform import (hemispherical_embed, build_s2h_basis_exact,
                            s2h_transform, s2h_inverse, project_grid_to_pixels,
                            s2h_degree_indices)
from s2h_localization import calculate_entropy_uncertainty, calculate_s2h_novelty_score


def _bbox_from_pixel_heatmap(pixel_heatmap, percentile=92, blur_ksize=5):
    """
    Percentile-based threshold instead of a fixed fraction of the map's
    max value. Fraction-of-max fails when the anomaly map is broad/smooth
    (common at low harmonic degree) -- it still captures a huge area
    since the whole map stays close to its own max. Percentile
    thresholding instead keeps only the actual TOP percentile fraction of
    pixels, regardless of how peaked or flat the overall map is.

    Before thresholding, the heatmap is Gaussian-blurred and the resulting
    mask is morphologically cleaned (close then open) -- this is what
    turns a noisy/jagged pixel-level anomaly pattern into one clean,
    tight, single-blob bounding box instead of a ragged outline or
    scattered fragments.
    """
    if pixel_heatmap.max() <= 1e-8:
        return None

    smoothed = cv2.GaussianBlur(pixel_heatmap.astype(np.float32), (blur_ksize, blur_ksize), 0)

    thresh = np.percentile(smoothed, percentile)
    if thresh <= 1e-8:
        return None
    mask = (smoothed >= thresh).astype(np.uint8) * 255

    # close small gaps inside the blob, then remove small noise specks
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Pick the contour containing the GLOBAL PEAK of the heatmap, not the
    # largest-area one. When multiple separate hot regions clear the
    # threshold, a broad-but-mild blob can have more pixels than a small
    # but genuinely more anomalous one -- picking by area then follows
    # the wrong spot. Anchoring to the peak keeps the box on the single
    # most anomalous location, which is what "localize the tumor" means.
    peak_idx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    peak_y, peak_x = float(peak_idx[0]), float(peak_idx[1])

    chosen = None
    for c in contours:
        if cv2.pointPolygonTest(c, (peak_x, peak_y), False) >= 0:
            chosen = c
            break

    if chosen is None:
        # rare edge case: peak pixel didn't survive morphological cleanup.
        # Fall back to the contour with the highest MEAN intensity inside
        # its bounding box (still intensity-based, not just area-based).
        best_mean = -1.0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            region_mean = smoothed[y:y + h, x:x + w].mean()
            if region_mean > best_mean:
                best_mean = region_mean
                chosen = c

    x, y, w, h = cv2.boundingRect(chosen)
    return x, y, w, h


_LOGIT_MODEL_CACHE = {}


def _get_logit_model(model):
    """
    Builds (and caches) a version of the model that outputs PRE-SOFTMAX
    logits instead of post-softmax probabilities. Standard Grad-CAM
    practice: a confident/saturated softmax output (e.g. [0,0,0,1]) has
    gradient ~0 w.r.t. its inputs even though the underlying logit still
    varies -- computing gradients on the probability silently produces an
    all-zero heatmap for exactly the confident predictions you most want
    to explain. Using the logit avoids this.
    """
    key = id(model)
    if key in _LOGIT_MODEL_CACHE:
        return _LOGIT_MODEL_CACHE[key]

    final_layer = model.get_layer("predictions")
    W, b = final_layer.get_weights()
    pre_final_model = tf.keras.Model(model.input, final_layer.input)

    def logit_fn(inputs):
        features = pre_final_model(inputs)
        return tf.matmul(features, W) + b

    _LOGIT_MODEL_CACHE[key] = logit_fn
    return logit_fn


def run_s2h_pipeline(model, image_path, class_names, healthy_mean_coeffs, healthy_std_coeffs,
                      image_size=128, n_theta=24, n_phi=48, max_degree=6,
                      entropy_weight=0.6, bbox_percentile=92,
                      novelty_scale=3.0,
                      min_degree_for_localization=2, heatmap_blur_ksize=3,
                      theta_sector=None, phi_sector=(0.0, 2 * np.pi)):
    """
    Runs the full joint classification + localization + uncertainty
    pipeline on a single image path. Returns a dict the UI/panel renderer
    needs.

    min_degree_for_localization: harmonic degrees BELOW this are excluded
        when building the localization heatmap (but NOT for classification
        or uncertainty, which still use the full coefficient vector).
        Degrees 0-1 capture broad, whole-brain-scale variation (overall
        brightness, gross shape/tilt) -- letting those dominate the
        residual map is what was causing bounding boxes to cover most of
        the brain. Degree >=2 captures more localized spatial structure.
    """
    pil_img = Image.open(image_path).convert("L").resize((image_size, image_size))
    gray = np.array(pil_img)

    # CRITICAL: must match train_hybrid_s2h.py's apply_clahe() exactly.
    # Training preprocesses every image with CLAHE before computing S2H
    # coefficients AND before feeding it to the CNN branch. Without this,
    # inference sees a differently-distributed image than the model was
    # trained on -- degrading classification confidence AND making the
    # S2H coefficients look anomalous relative to the healthy profile
    # (since that profile was also computed on CLAHE images), which is
    # what was producing the noisy, scattered heatmap.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    orig_np = np.stack([gray, gray, gray], axis=-1).astype("uint8")

    # --- S2H transform (shared by classification, localization, uncertainty) ---
    I_grid, theta_grid, phi_grid, centroid, R_of_phi = hemispherical_embed(
        gray, n_theta=n_theta, n_phi=n_phi
    )
    sector = theta_sector if theta_sector is not None else (theta_grid[0], theta_grid[-1])
    basis, weights, _ = build_s2h_basis_exact(theta_grid, phi_grid, max_degree, sector, phi_sector)
    coeffs = s2h_transform(I_grid, basis, weights)

    # --- Classification (hybrid model: image + coefficients) ---
    img_batch = np.expand_dims(orig_np.astype("float32") / 255.0, axis=0)
    coeffs_batch = np.expand_dims(coeffs.astype("float32"), axis=0)
    pred_probs = model.predict([img_batch, coeffs_batch], verbose=0)[0]

    class_idx = int(np.argmax(pred_probs))
    predicted_class = class_names[class_idx]
    confidence = float(pred_probs[class_idx])

    # --- Localization: S2H-CAM -- gradient of the predicted class score
    # w.r.t. the S2H coefficients, the harmonic-domain analogue of
    # Grad-CAM. This ties the heatmap directly to what the CLASSIFIER
    # actually used to make its decision, instead of comparing against a
    # population-average "healthy" profile (which flags ANY anatomical
    # variation -- asymmetry, ventricle size, artifacts -- not just the
    # tumor, and was producing the scattered multi-spot heatmaps).
    img_tensor = tf.convert_to_tensor(img_batch, dtype=tf.float32)
    coeffs_tensor = tf.convert_to_tensor(coeffs_batch, dtype=tf.float32)
    logit_fn = _get_logit_model(model)
    with tf.GradientTape() as tape:
        tape.watch(coeffs_tensor)
        logits = logit_fn([img_tensor, coeffs_tensor])
        class_logit = logits[:, class_idx]
    grads = tape.gradient(class_logit, coeffs_tensor)[0].numpy()

    # Grad-CAM-style: positive gradients only (coefficients that PUSH
    # TOWARD the predicted class), weighted by how strongly each
    # coefficient is actually expressed in this image.
    importance = np.maximum(grads, 0.0) * np.abs(coeffs)
    # low-degree terms still carry mostly global/whole-brain information;
    # de-emphasizing them keeps the map focused on localized structure
    degree_idx = s2h_degree_indices(max_degree)
    importance = np.where(degree_idx >= min_degree_for_localization, importance, 0.0)

    cam_grid = s2h_inverse(importance, basis, I_grid.shape)
    cam_grid = np.maximum(cam_grid, 0.0)  # ReLU, same as Grad-CAM
    pixel_heatmap = project_grid_to_pixels(
        cam_grid, theta_grid, phi_grid, centroid, R_of_phi, gray.shape
    )
    # project_grid_to_pixels is a nearest-neighbor lookup, which looks
    # blocky/pixelated at full image resolution -- light smoothing gives
    # a cleaner look without washing out the now much sharper signal.
    pixel_heatmap = cv2.GaussianBlur(pixel_heatmap.astype(np.float32), (heatmap_blur_ksize, heatmap_blur_ksize), 0)
    if pixel_heatmap.max() > 0:
        pixel_heatmap_norm = pixel_heatmap / pixel_heatmap.max()
    else:
        pixel_heatmap_norm = pixel_heatmap

    bbox = None
    if predicted_class != "notumor":
        bbox = _bbox_from_pixel_heatmap(pixel_heatmap_norm, percentile=bbox_percentile)

    # --- Uncertainty: entropy + S2H novelty ---
    certainty, uncertainty = calculate_entropy_uncertainty(pred_probs, len(class_names))
    novelty = calculate_s2h_novelty_score(coeffs, healthy_mean_coeffs, healthy_std_coeffs, scale=novelty_scale)
    combined_uncertainty = entropy_weight * uncertainty + (1 - entropy_weight) * novelty
    combined_certainty = 1 - combined_uncertainty

    confidence_status = (
        "High Confidence" if confidence > 0.85 and combined_uncertainty < 0.3
        else "Moderate Confidence" if confidence > 0.6
        else "Low Confidence"
    )

    return {
        "orig_np": orig_np,
        "heatmap_norm": pixel_heatmap_norm,  # raw 0-1 map, for a matplotlib-native overlay + colorbar
        "bbox": bbox,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "certainty": combined_certainty,
        "uncertainty": combined_uncertainty,
        "confidence_status": confidence_status,
        "all_probs": {class_names[i]: float(pred_probs[i]) for i in range(len(class_names))},
    }


def render_panel(result, output_path):
    """
    Renders TWO panels to a PNG file:
      1. Original MRI
      2. S2H heatmap overlay WITH the localized-tumor bounding box drawn
         directly on top of it, plus a colorbar showing what the heatmap
         colors mean (S2H residual anomaly, relative to the learned
         healthy-brain harmonic profile: 0 = matches healthy profile,
         1 = maximum deviation seen in this image).
    No title, no prediction text baked in -- that's shown in the HTML page.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    orig_np = result["orig_np"]
    heatmap_norm = result["heatmap_norm"]
    bbox = result["bbox"]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6))

    axes[0].imshow(orig_np)
    axes[0].set_title("1. Original MRI")
    axes[0].axis("off")

    axes[1].imshow(orig_np)
    heat_im = axes[1].imshow(heatmap_norm, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    heatmap_title = (
        "2. S2H Attention (No Focal Anomaly)" if result["predicted_class"] == "notumor"
        else "2. S2H Heatmap — Localized Tumor"
    )
    axes[1].set_title(heatmap_title)
    axes[1].axis("off")

    if bbox is not None:
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2.5, edgecolor="white", facecolor="none")
        axes[1].add_patch(rect)
        label_y = y - 6
        va = "bottom"
        if label_y < 8:
            label_y = y + 6
            va = "top"
        axes[1].text(
            x, label_y, "Tumor",
            color="white", fontsize=10, fontweight="bold", va=va, ha="left",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", edgecolor="none", alpha=0.7),
        )

    cbar = fig.colorbar(heat_im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("S2H Anomaly vs. Healthy Profile", fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Medium", "High"])
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
