import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from preprocessing.pso_he import apply_pso_he


def load_images_and_masks(
    train_path,
    image_size=256,
    apply_pso=True,
    image_folder="images",
    mask_folder="masks"
):
    image_paths = sorted(glob.glob(os.path.join(train_path, image_folder, "**", "*.png"), recursive=True))
    mask_paths = sorted(glob.glob(os.path.join(train_path, mask_folder, "**", "*.png"), recursive=True))

    if len(image_paths) != len(mask_paths):
        raise ValueError("Number of images and masks does not match.")

    images = []
    masks = []

    for img_path, mask_path in zip(image_paths, mask_paths):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")

        img = cv2.resize(img, (image_size, image_size))

        if apply_pso:
            img, _ = apply_pso_he(img)

        img = img.astype(np.uint8)
        img = cv2.equalizeHist(img)
        img = cv2.merge([img, img, img])

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise ValueError(f"Cannot read mask: {mask_path}")

        mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask / 255.0).astype(np.uint8)

        images.append(img)
        masks.append(mask)

    images = np.array(images, dtype=np.uint8)
    masks = np.array(masks, dtype=np.uint8)

    return images, masks, image_paths, mask_paths


def create_dataset(images, masks, batch_size=8, shuffle=True):
    images = images.astype("float32") / 255.0
    masks = masks.astype("float32")

    if masks.ndim == 3:
        masks = np.expand_dims(masks, axis=-1)

    dataset = tf.data.Dataset.from_tensor_slices((images, masks))

    if shuffle:
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def split_dataset(images, masks, test_size=0.2, val_size=0.5, seed=42):
    train_images, temp_images, train_masks, temp_masks = train_test_split(
        images, masks, test_size=test_size, random_state=seed
    )

    valid_images, test_images, valid_masks, test_masks = train_test_split(
        temp_images, temp_masks, test_size=val_size, random_state=seed
    )

    return train_images, valid_images, test_images, train_masks, valid_masks, test_masks
