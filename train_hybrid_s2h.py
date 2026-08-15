"""
train_hybrid_s2h.py
====================
Focused training script: HYBRID CNN+S2H model only (no CNN-only / pure-S2H
ablation runs). Includes CLAHE preprocessing + augmentation to match your
92%-accuracy Training.ipynb baseline, so the hybrid model has a fair shot
at matching or beating it.

USAGE
-----
    python train_hybrid_s2h.py \
        --train-dir Dataset/Training --test-dir Dataset/Testing \
        --epochs 30

Saves to --out-dir: hybrid.keras, inference_config.json, class_names.json,
healthy_mean_coeffs.npy, healthy_std_coeffs.npy -- everything main.py needs.
"""

import argparse
import os
import json
import numpy as np
import cv2
from PIL import Image

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from s2h_transform import extract_s2h_features, calibrate_sector_from_data, compute_q1_q2
from s2h_localization import calibrate_novelty_scale
from s2h_models import build_hybrid_model


def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def augment_image(gray, training=True):
    if not training:
        return gray
    img = gray.astype(np.float32)
    if np.random.rand() < 0.5:
        alpha = np.random.uniform(0.9, 1.1)   # contrast
        beta = np.random.uniform(-10, 10)     # brightness
        img = img * alpha + beta
    return np.clip(img, 0, 255).astype(np.uint8)


def load_dataset(data_dir, class_names, image_size, max_degree, n_theta, n_phi,
                  max_images_per_class=None, training=False,
                  theta_sector=None, phi_sector=(0.0, 2 * np.pi)):
    images, coeffs_list, labels = [], [], []

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"WARNING: {class_dir} not found, skipping")
            continue
        files = sorted(os.listdir(class_dir))
        if max_images_per_class:
            files = files[:max_images_per_class]

        for fname in files:
            fpath = os.path.join(class_dir, fname)
            try:
                pil_img = Image.open(fpath).convert("L").resize((image_size, image_size))
            except Exception as e:
                print(f"Skipping unreadable file {fpath}: {e}")
                continue
            gray = np.array(pil_img)
            gray = apply_clahe(gray)
            gray = augment_image(gray, training=training)

            coeffs, _ = extract_s2h_features(
                gray, n_theta=n_theta, n_phi=n_phi, max_degree=max_degree,
                theta_sector=theta_sector, phi_sector=phi_sector,
            )
            rgb = np.stack([gray, gray, gray], axis=-1).astype("float32") / 255.0

            images.append(rgb)
            coeffs_list.append(coeffs)
            labels.append(class_idx)

    X_img = np.array(images, dtype="float32")
    X_coeffs = np.array(coeffs_list, dtype="float32")
    y = tf.keras.utils.to_categorical(labels, num_classes=len(class_names))
    return X_img, X_coeffs, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--class-names", nargs="+",
                         default=["glioma", "meningioma", "notumor", "pituitary"])
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-degree", type=int, default=6)
    parser.add_argument("--n-theta", type=int, default=24)
    parser.add_argument("--n-phi", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-images-per-class", type=int, default=None)
    parser.add_argument("--calibration-samples", type=int, default=200,
                         help="How many training images (spread across classes) to use "
                              "for CALCULATING the S2H sector bounds via variance concentration.")
    parser.add_argument("--variance-capture", type=float, default=0.85,
                         help="Fraction of population variance the calculated sector must capture.")
    parser.add_argument("--out-dir", default="./s2h_hybrid_results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    class_names = args.class_names
    num_classes = len(class_names)

    # ---- CALCULATE the S2H sector from data (not assumed) ----
    print(f"Calibrating S2H sector from {args.calibration_samples} sample training images...")
    calib_images = []
    per_class = max(1, args.calibration_samples // len(class_names))
    for class_name in class_names:
        class_dir = os.path.join(args.train_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        files = sorted(os.listdir(class_dir))[:per_class]
        for fname in files:
            try:
                pil_img = Image.open(os.path.join(class_dir, fname)).convert("L").resize(
                    (args.image_size, args.image_size)
                )
                gray = apply_clahe(np.array(pil_img))
                calib_images.append(gray)
            except Exception:
                continue

    theta_sector, phi_sector, var_map = calibrate_sector_from_data(
        calib_images, n_theta=args.n_theta, n_phi=args.n_phi,
        variance_capture=args.variance_capture,
    )
    q1, q2 = compute_q1_q2(*theta_sector)
    u = 2 * np.pi / (phi_sector[1] - phi_sector[0])
    print(f"Calculated sector: theta=({np.rad2deg(theta_sector[0]):.1f}, "
          f"{np.rad2deg(theta_sector[1]):.1f}) deg, phi=({np.rad2deg(phi_sector[0]):.1f}, "
          f"{np.rad2deg(phi_sector[1]):.1f}) deg")
    print(f"Derived q1={q1:.4f}, q2={q2:.4f}, u={u:.4f}")
    np.save(os.path.join(args.out_dir, "variance_map.npy"), var_map)

    print("Loading + computing S2H features for training set (CLAHE + augment)...")
    X_img_train, X_coeffs_train, y_train = load_dataset(
        args.train_dir, class_names, args.image_size, args.max_degree,
        args.n_theta, args.n_phi, args.max_images_per_class, training=True,
        theta_sector=theta_sector, phi_sector=phi_sector,
    )
    print("Loading + computing S2H features for test set (CLAHE, no augment)...")
    X_img_test, X_coeffs_test, y_test = load_dataset(
        args.test_dir, class_names, args.image_size, args.max_degree,
        args.n_theta, args.n_phi, args.max_images_per_class, training=False,
        theta_sector=theta_sector, phi_sector=phi_sector,
    )

    num_coeffs = X_coeffs_train.shape[1]
    print(f"Train: {X_img_train.shape[0]} images | Test: {X_img_test.shape[0]} images | "
          f"S2H coeffs per image: {num_coeffs}")

    healthy_mean_coeffs, healthy_std_coeffs = None, None
    if "notumor" in class_names:
        notumor_idx = class_names.index("notumor")
        healthy_mask = np.argmax(y_train, axis=1) == notumor_idx
        healthy_coeffs = X_coeffs_train[healthy_mask]
        healthy_mean_coeffs = healthy_coeffs.mean(axis=0)
        healthy_std_coeffs = healthy_coeffs.std(axis=0)
        np.save(os.path.join(args.out_dir, "healthy_mean_coeffs.npy"), healthy_mean_coeffs)
        np.save(os.path.join(args.out_dir, "healthy_std_coeffs.npy"), healthy_std_coeffs)

    # CALIBRATE the novelty scale from data (not assumed). Compute raw
    # z-score energy across ALL training images (every class, not just
    # healthy ones -- we want to know what "typical" looks like across
    # the whole population), then pick a scale so the 90th percentile of
    # that reads as ~30% novel by default. This is what stops correctly,
    # confidently classified normal images from reading as ~70% novel
    # just because the squashing constant was never fit to real data.
    novelty_scale = 3.0
    if healthy_mean_coeffs is not None:
        z_all = (X_coeffs_train - healthy_mean_coeffs) / (healthy_std_coeffs + 1e-8)
        raw_scores_all = np.sqrt(np.mean(z_all ** 2, axis=1))
        novelty_scale = calibrate_novelty_scale(raw_scores_all, target_novelty_at_p90=0.3)
        print(f"Calibrated novelty scale: {novelty_scale:.4f} "
              f"(raw score p90={np.percentile(raw_scores_all,90):.4f})")

    # save inference artifacts main.py needs -- including the CALCULATED sector
    inference_config = {
        "class_names": class_names, "image_size": args.image_size,
        "n_theta": args.n_theta, "n_phi": args.n_phi,
        "max_degree": args.max_degree, "num_coeffs": int(num_coeffs),
        "theta_sector": list(theta_sector), "phi_sector": list(phi_sector),
        "q1": q1, "q2": q2, "u": u,
        "novelty_scale": novelty_scale,
    }
    with open(os.path.join(args.out_dir, "inference_config.json"), "w") as f:
        json.dump(inference_config, f, indent=2)

    # stratified train/val split (avoids the class-ordered-data bug)
    y_labels = np.argmax(y_train, axis=1)
    idx = np.arange(len(y_labels))
    idx_tr, idx_val = train_test_split(idx, test_size=0.15, random_state=42, stratify=y_labels)

    model = build_hybrid_model(args.image_size, num_coeffs, num_classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.out_dir, "hybrid.keras"), monitor="val_accuracy", save_best_only=True
        ),
    ]

    model.fit(
        [X_img_train[idx_tr], X_coeffs_train[idx_tr]], y_train[idx_tr],
        validation_data=([X_img_train[idx_val], X_coeffs_train[idx_val]], y_train[idx_val]),
        epochs=args.epochs, batch_size=32, callbacks=callbacks, verbose=1,
    )

    preds = model.predict([X_img_test, X_coeffs_test], verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\n=== HYBRID CNN+S2H RESULTS ===")
    print(classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    with open(os.path.join(args.out_dir, "hybrid_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    np.save(os.path.join(args.out_dir, "hybrid_confusion_matrix.npy"), cm)

    print(f"\nFinal test accuracy: {report['accuracy']*100:.2f}%")
    print(f"Saved model + artifacts to: {args.out_dir}")


if __name__ == "__main__":
    main()
