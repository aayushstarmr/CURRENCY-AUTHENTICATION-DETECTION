"""
Indian Currency REAL/FAKE Detection with a custom Vision Transformer (ViT).

Run from your VS Code project directory:
    python main.py --dataset "$HOME/Downloads/v1.zip"

Or use an extracted dataset directory:
    python main.py --dataset "$HOME/Downloads/currency_dataset"

Folder names must identify images as REAL/FAKE (or genuine/forged), for example:
    dataset/real/10/note1.jpg
    dataset/fake/10/note2.jpg

This code measures real validation/test performance. No script can guarantee
89% accuracy on a particular dataset; results depend on its size, labels, and quality.
"""
import argparse
import os
import random
import zipfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

ImageFile.LOAD_TRUNCATED_IMAGES = True
SEED = 42
IMAGE_SIZE = 224
PATCH_SIZE = 16
BATCH_SIZE = 16
DEFAULT_EPOCHS = 50
AUTOTUNE = tf.data.AUTOTUNE
CLASS_NAMES = ["FAKE", "REAL"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def infer_label(path: Path):
    """Infer binary label from path folder names."""
    parts = [part.lower().replace("-", "_").replace(" ", "_") for part in path.parts]
    fake_words = ("fake", "forged", "counterfeit", "fraud", "not_genuine", "not_real")
    real_words = ("real", "genuine", "authentic")
    # Use the closest matching directory name first.
    for part in reversed(parts[:-1]):
        if any(word in part for word in fake_words):
            return 0
        if any(word in part for word in real_words):
            return 1
    return None


def prepare_dataset(dataset_arg: str, project_dir: Path):
    dataset_path = Path(dataset_arg).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    if dataset_path.is_file() and dataset_path.suffix.lower() == ".zip":
        extract_dir = project_dir / "dataset_extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        if not any(extract_dir.iterdir()):
            print(f"Extracting dataset ZIP: {dataset_path}")
            root = extract_dir.resolve()
            with zipfile.ZipFile(dataset_path) as zf:
                for member in zf.infolist():
                    target = (extract_dir / member.filename).resolve()
                    if target != root and root not in target.parents:
                        raise ValueError("Unsafe path found in ZIP archive.")
                zf.extractall(extract_dir)
        search_root = extract_dir
    elif dataset_path.is_dir():
        search_root = dataset_path
    else:
        raise ValueError("--dataset must point to a ZIP file or extracted folder.")

    image_paths, labels = [], []
    for image_path in search_root.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = infer_label(image_path)
        if label is None:
            continue
        try:
            with Image.open(image_path) as im:
                im.verify()
            image_paths.append(str(image_path))
            labels.append(label)
        except Exception:
            print(f"Skipping unreadable image: {image_path}")

    if not image_paths:
        folder_sample = [str(p.relative_to(search_root)) for p in search_root.rglob("*") if p.is_dir()]
        raise RuntimeError(
            "No labelled REAL/FAKE images found. Folder names must include 'real'/'genuine' "
            "or 'fake'/'forged'. Example folders found:\n" + "\n".join(folder_sample[:30])
        )

    labels = np.asarray(labels, dtype=np.int32)
    print(f"Found {len(image_paths)} labelled images.")
    print(f"FAKE: {int(np.sum(labels == 0))} | REAL: {int(np.sum(labels == 1))}")
    if len(np.unique(labels)) != 2:
        raise RuntimeError("Both REAL and FAKE classes must be present.")
    if min(np.sum(labels == 0), np.sum(labels == 1)) < 5:
        raise RuntimeError("Need at least 5 images in each class to create train/validation/test splits.")
    return np.asarray(image_paths), labels


def load_image(path, label):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.cast(label, tf.float32)


def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, projection_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = tf.keras.layers.Conv2D(
            projection_dim, kernel_size=patch_size, strides=patch_size, padding="valid"
        )
        self.position_embedding = tf.keras.layers.Embedding(self.num_patches, projection_dim)

    def call(self, images):
        x = self.projection(images)
        x = tf.reshape(x, [tf.shape(x)[0], self.num_patches, -1])
        positions = tf.range(self.num_patches)
        return x + self.position_embedding(positions)


def transformer_block(x, projection_dim=128, num_heads=4, mlp_dim=256, dropout=0.15):
    y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim // num_heads, dropout=dropout
    )(y, y)
    y = tf.keras.layers.Dropout(dropout)(y)
    x = tf.keras.layers.Add()([x, y])

    y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    y = tf.keras.layers.Dense(mlp_dim, activation=tf.nn.gelu)(y)
    y = tf.keras.layers.Dropout(dropout)(y)
    y = tf.keras.layers.Dense(projection_dim)(y)
    y = tf.keras.layers.Dropout(dropout)(y)
    return tf.keras.layers.Add()([x, y])


def build_vit():
    inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3))
    x = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.035),
        tf.keras.layers.RandomZoom(0.07),
        tf.keras.layers.RandomContrast(0.10),
    ], name="training_augmentation")(inputs)
    x = PatchEmbedding()(x)
    for _ in range(6):
        x = transformer_block(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(128, activation=tf.nn.gelu)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="real_probability")(x)

    model = tf.keras.Model(inputs, outputs, name="currency_vit")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model


def predict_one(model, image_path, output_dir=None):
    image_path = Path(image_path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    original = Image.open(image_path).convert("RGB")
    resized = original.resize((IMAGE_SIZE, IMAGE_SIZE))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    p_real = float(model.predict(array[None, ...], verbose=0)[0][0])
    label = "REAL" if p_real >= 0.5 else "FAKE"
    confidence = p_real if label == "REAL" else 1.0 - p_real
    plt.figure(figsize=(6, 6))
    plt.imshow(original)
    plt.title(f"Prediction: {label} | Model confidence: {confidence * 100:.1f}%")
    plt.axis("off")
    plt.tight_layout()
    if output_dir:
        plt.savefig(Path(output_dir) / "prediction.png", dpi=160, bbox_inches="tight")
    plt.show()
    print(f"Prediction: {label} | Model confidence: {confidence * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Train a ViT for Indian currency REAL/FAKE classification.")
    parser.add_argument("--dataset", required=True, help="Path to dataset ZIP or extracted folder")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--image", default=None, help="Optional image path to predict after training")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths, labels = prepare_dataset(args.dataset, project_dir)
    train_paths, temp_paths, train_y, temp_y = train_test_split(
        paths, labels, test_size=0.30, random_state=SEED, stratify=labels
    )
    val_paths, test_paths, val_y, test_y = train_test_split(
        temp_paths, temp_y, test_size=0.50, random_state=SEED, stratify=temp_y
    )
    print(f"Train: {len(train_paths)} | Validation: {len(val_paths)} | Test: {len(test_paths)}")

    train_ds = make_dataset(train_paths, train_y, training=True)
    val_ds = make_dataset(val_paths, val_y)
    test_ds = make_dataset(test_paths, test_y)

    model = build_vit()
    model.summary()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(output_dir / "best_currency_vit.keras"),
            monitor="val_accuracy", mode="max", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "training_log.csv"))
    ]

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks
    )
    model.save(output_dir / "currency_vit_final.keras")

    # Evaluate the restored best validation checkpoint on held-out test data.
    best_path = output_dir / "best_currency_vit.keras"
    if best_path.exists():
        model = tf.keras.models.load_model(
            best_path, custom_objects={"PatchEmbedding": PatchEmbedding}
        )
    test_probs = model.predict(test_ds, verbose=1).reshape(-1)
    test_preds = (test_probs >= 0.5).astype(np.int32)
    print(f"\nHeld-out TEST accuracy: {accuracy_score(test_y, test_preds) * 100:.2f}%")
    print(classification_report(
        test_y, test_preds, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0
    ))

    ConfusionMatrixDisplay(
        confusion_matrix(test_y, test_preds, labels=[0, 1]), display_labels=CLASS_NAMES
    ).plot(values_format="d")
    plt.title("ViT Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=160)
    plt.show()

    if args.image:
        predict_one(model, args.image, output_dir)


if __name__ == "__main__":
    main()
