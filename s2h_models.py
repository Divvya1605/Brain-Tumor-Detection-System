"""
s2h_models.py
=============
Two classifier architectures for the ablation study:

  1. build_pure_s2h_model  -- classification from S2H coefficients ALONE
     (no CNN, no raw pixels). Tests whether the harmonic-domain
     representation by itself carries enough signal to classify tumors.

  2. build_hybrid_model    -- your existing VGG16 branch (GAP features)
     CONCATENATED with the S2H coefficient vector, feeding a shared
     classification head. Tests whether S2H features add information the
     CNN doesn't already extract on its own.

Train both (plus your existing CNN-only baseline) on the same
train/val/test split for a clean 3-way ablation:
    CNN-only   vs   Pure-S2H   vs   Hybrid CNN+S2H
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16


# ---------------------------------------------------------------------------
# 1. Pure S2H model (coefficients only)
# ---------------------------------------------------------------------------

def build_pure_s2h_model(num_coeffs, num_classes=4, dropout=0.3):
    """
    A small dense network operating purely on S2H coefficient vectors.
    Input shape: (num_coeffs,) -- e.g. 49 for max_degree=6.
    """
    inputs = layers.Input(shape=(num_coeffs,), name="s2h_coeffs")
    x = layers.BatchNormalization()(inputs)  # coefficients have very different scales per degree
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name="pure_s2h_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# 2. Hybrid CNN + S2H model
# ---------------------------------------------------------------------------

def build_hybrid_model(image_size=128, num_coeffs=49, num_classes=4,
                        dropout=0.3, fine_tune_from_block5=True):
    """
    Two-input model:
      - image_input  -> VGG16 (ImageNet weights, block5 fine-tunable) -> GAP -> 512-d
      - coeffs_input -> BatchNorm -> Dense(64) S2H branch
      concatenated -> Dense head -> softmax

    This mirrors your existing Training.ipynb CNN architecture (VGG16 +
    GAP) so the CNN branch is a fair, direct comparison point, with the
    S2H branch added on top.
    """
    image_input = layers.Input(shape=(image_size, image_size, 3), name="image_input")
    coeffs_input = layers.Input(shape=(num_coeffs,), name="s2h_coeffs")

    base_model = VGG16(weights="imagenet", include_top=False,
                        input_shape=(image_size, image_size, 3))
    base_model.trainable = True
    for layer in base_model.layers:
        if fine_tune_from_block5:
            layer.trainable = layer.name.startswith("block5")
        else:
            layer.trainable = False

    cnn_features = base_model(image_input)
    cnn_features = layers.GlobalAveragePooling2D(name="cnn_gap")(cnn_features)

    s2h_branch = layers.BatchNormalization()(coeffs_input)
    s2h_branch = layers.Dense(64, activation="relu", name="s2h_dense")(s2h_branch)

    merged = layers.Concatenate(name="cnn_s2h_concat")([cnn_features, s2h_branch])
    x = layers.Dropout(dropout)(merged)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model([image_input, coeffs_input], outputs, name="hybrid_cnn_s2h_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
