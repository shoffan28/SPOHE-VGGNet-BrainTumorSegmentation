import os, cv2
import numpy as np
import matplotlib.pyplot as plt

def create_error_heatmap(image, ground_truth, prediction, alpha=0.4):
    gt = (ground_truth.squeeze() > 0).astype(np.uint8)
    pred = (prediction.squeeze() > 0.5).astype(np.uint8)

    error_map = np.abs(gt - pred).astype(np.uint8) * 255

    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(error_map, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image.astype(np.uint8), 1 - alpha, heatmap, alpha, 0)
    return overlay, error_map


def save_error_maps(test_ds, model, output_dir="visualization/error_maps", n_samples=10):
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for image_batch, mask_batch in test_ds:
        preds = model.predict(image_batch, verbose=0)

        for i in range(image_batch.shape[0]):
            image = image_batch[i].numpy()
            mask = mask_batch[i].numpy()
            pred = preds[i]

            overlay, error_map = create_error_heatmap(image, mask, pred)

            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title("MRI")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(error_map, cmap="hot")
            plt.title("Error Map")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title("Overlay")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"error_map_{count+1}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()

            count += 1
            if count >= n_samples:
                return
