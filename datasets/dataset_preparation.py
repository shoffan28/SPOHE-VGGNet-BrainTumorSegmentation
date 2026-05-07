import os, glob, cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from preprocessing.pso_he import apply_pso_he

def load_images_masks(root_dir, image_size=256, apply_pso=True):
    image_paths = sorted(glob.glob(os.path.join(root_dir, "images", "**", "*.png"), recursive=True))
    mask_paths = sorted(glob.glob(os.path.join(root_dir, "masks", "**", "*.png"), recursive=True))

    images, masks = [], []

    for img_path, mask_path in zip(image_paths, mask_paths):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (image_size, image_size))

        if apply_pso:
            img, _ = apply_pso_he(img)

        img = cv2.equalizeHist(img.astype(np.uint8))
        img = cv2.merge([img, img, img])

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask / 255.0).astype(np.uint8)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks), image_paths, mask_paths


def split_data(images, masks, seed=42):
    train_x, temp_x, train_y, temp_y = train_test_split(
        images, masks, test_size=0.2, random_state=seed
    )
    val_x, test_x, val_y, test_y = train_test_split(
        temp_x, temp_y, test_size=0.5, random_state=seed
    )
    return train_x, val_x, test_x, train_y, val_y, test_y


def create_tf_dataset(images, masks, batch_size=8, shuffle=True):
    images = images.astype("float32") / 255.0
    masks = masks.astype("float32")

    if masks.ndim == 3:
        masks = np.expand_dims(masks, axis=-1)

    ds = tf.data.Dataset.from_tensor_slices((images, masks))
    if shuffle:
        ds = ds.shuffle(1000)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
