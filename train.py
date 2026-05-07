import os
import tensorflow as tf

from utils.data_loader import load_images_and_masks, create_dataset, split_dataset
from utils.metrics import dice_coef, dice_loss, jaccard_index, precision_metric, recall_metric, f1_metric
from models.spohe_vggnet import build_spohe_vggnet


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

TRAIN_PATH = "/net/pr2/projects/plgrid/plggman01/Brain Tumor Segmentation"
IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-3


def main():
    images, masks, _, _ = load_images_and_masks(
        train_path=TRAIN_PATH,
        image_size=IMAGE_SIZE,
        apply_pso=True
    )

    train_images, valid_images, test_images, train_masks, valid_masks, test_masks = split_dataset(
        images, masks
    )

    train_ds = create_dataset(train_images, train_masks, batch_size=BATCH_SIZE, shuffle=True)
    valid_ds = create_dataset(valid_images, valid_masks, batch_size=BATCH_SIZE, shuffle=False)
    test_ds = create_dataset(test_images, test_masks, batch_size=BATCH_SIZE, shuffle=False)

    model = build_spohe_vggnet(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        dropout_rate=0.1
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=[
            "binary_accuracy",
            dice_coef,
            jaccard_index,
            precision_metric,
            recall_metric,
            f1_metric
        ]
    )

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir="logs"),
        tf.keras.callbacks.ModelCheckpoint(
            "best_spohe_vggnet.h5",
            monitor="val_dice_coef",
            mode="max",
            save_best_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    results = model.evaluate(test_ds)
    print("Test results:", results)

    model.save("spohe_vggnet_final.h5")


if __name__ == "__main__":
    main()
