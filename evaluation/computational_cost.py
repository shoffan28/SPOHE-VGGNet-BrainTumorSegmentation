import time
import numpy as np
import pandas as pd
import tensorflow as tf


class EpochTimeCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.time() - self.epoch_start)


def count_parameters_million(model):
    return model.count_params() / 1e6


def measure_inference_time(model, test_ds, warmup_batches=3):
    for i, (image_batch, _) in enumerate(test_ds):
        _ = model.predict(image_batch, verbose=0)
        if i + 1 >= warmup_batches:
            break

    total_time = 0.0
    total_images = 0

    for image_batch, _ in test_ds:
        start = time.time()
        _ = model.predict(image_batch, verbose=0)
        end = time.time()

        total_time += end - start
        total_images += image_batch.shape[0]

    return (total_time / total_images) * 1000


def save_computational_profile(
    model,
    test_ds,
    avg_epoch_time,
    total_training_time,
    psohe_time_ms,
    output_csv="computational_profile.csv"
):
    inference_time_ms = measure_inference_time(model, test_ds)
    end_to_end_time = inference_time_ms + psohe_time_ms

    result = {
        "Model": "SPOHE-VGGNet",
        "Parameters (M)": round(count_parameters_million(model), 3),
        "Training Time/Epoch (s)": round(avg_epoch_time, 3),
        "Total Training Time (s)": round(total_training_time, 3),
        "Inference Time/Image (ms)": round(inference_time_ms, 3),
        "PSO-HE Preprocessing Time/Image (ms)": round(psohe_time_ms, 3),
        "End-to-End Time/Image (ms)": round(end_to_end_time, 3)
    }

    df = pd.DataFrame([result])
    df.to_csv(output_csv, index=False)

    print(df)
    return df
