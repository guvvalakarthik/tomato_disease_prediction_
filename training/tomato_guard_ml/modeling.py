from __future__ import annotations

from typing import Any


def build_mobilenet_model(
    image_size: tuple[int, int], class_count: int, seed: int
) -> tuple[Any, Any]:
    import tensorflow as tf

    inputs = tf.keras.Input((*image_size, 3), name="image", dtype=tf.float32)
    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.12, seed=seed + 1),
            tf.keras.layers.RandomZoom(0.15, seed=seed + 2),
            tf.keras.layers.RandomTranslation(0.1, 0.1, seed=seed + 3),
            tf.keras.layers.RandomContrast(0.2, seed=seed + 4),
        ],
        name="field_augmentation",
    )(inputs)
    backbone = tf.keras.applications.MobileNetV3Small(
        input_shape=(*image_size, 3),
        include_top=False,
        include_preprocessing=True,
        weights="imagenet",
        pooling=None,
    )
    backbone.trainable = False
    features = tf.keras.layers.Activation("linear", name="features")(backbone(augment))
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(features)
    pooled = tf.keras.layers.Dropout(0.25, seed=seed, name="dropout")(pooled)
    logits = tf.keras.layers.Dense(class_count, name="classifier")(pooled)
    return tf.keras.Model(inputs, logits, name="tomatoguard_mobilenetv3"), backbone


def build_baseline_model(
    image_size: tuple[int, int], class_count: int, seed: int
) -> tuple[Any, None]:
    import tensorflow as tf

    inputs = tf.keras.Input((*image_size, 3), name="image", dtype=tf.float32)
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    x = tf.keras.layers.RandomFlip("horizontal", seed=seed)(x)
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    features = tf.keras.layers.Activation("linear", name="features")(x)
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(features)
    logits = tf.keras.layers.Dense(class_count, name="classifier")(pooled)
    return tf.keras.Model(inputs, logits, name="tomatoguard_baseline"), None


def compile_model(model: Any, learning_rate: float) -> None:
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def unfreeze_backbone(backbone: Any, frozen_fraction: float = 0.70) -> None:
    if backbone is None:
        return
    backbone.trainable = True
    cutoff = int(len(backbone.layers) * frozen_fraction)
    for layer in backbone.layers[:cutoff]:
        layer.trainable = False
    for layer in backbone.layers[cutoff:]:
        if layer.__class__.__name__ == "BatchNormalization":
            layer.trainable = False
