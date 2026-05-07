import os, cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def make_gradcam_heatmap(img_array, model, layer_name="block5_conv4"):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = tf.reduce_mean(predictions)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_heatmap(image, heatmap, alpha=0.45):
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)


def save_gradcam_samples(test_ds, model, output_dir="visualization/gradcam_outputs",
                         n_samples=5, layer_name="block5_conv4"):
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for image_batch, mask_batch in test_ds:
        preds = model.predict(image_batch, verbose=0)

        for i in range(image_batch.shape[0]):
            image = image_batch[i].numpy()
            mask = mask_batch[i].numpy()
            pred = preds[i]
            pred_binary = (pred > 0.5).astype(np.uint8)

            heatmap = make_gradcam_heatmap(
                np.expand_dims(image, axis=0),
                model,
                layer_name
            )

            overlay = overlay_heatmap(image, heatmap)
            error_map = np.abs(mask.squeeze() - pred_binary.squeeze())

            plt.figure(figsize=(22, 5))

            plt.subplot(1, 5, 1)
            plt.imshow(image)
            plt.title("MRI")
            plt.axis("off")

            plt.subplot(1, 5, 2)
            plt.imshow(mask.squeeze(), cmap="gray")
            plt.title("Ground Truth")
            plt.axis("off")

            plt.subplot(1, 5, 3)
            plt.imshow(pred_binary.squeeze(), cmap="gray")
            plt.title("Prediction")
            plt.axis("off")

            plt.subplot(1, 5, 4)
            plt.imshow(overlay)
            plt.title("Grad-CAM")
            plt.axis("off")

            plt.subplot(1, 5, 5)
            plt.imshow(error_map, cmap="hot")
            plt.title("Error Map")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"gradcam_sample_{count+1}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()

            count += 1
            if count >= n_samples:
                return
