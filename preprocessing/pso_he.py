import cv2
import numpy as np
from scipy.optimize import minimize


def calculate_entropy(image: np.ndarray) -> float:
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    histogram = histogram / (np.sum(histogram) + 1e-9)
    entropy = -np.sum(histogram * np.log2(histogram + 1e-9))
    return entropy


def pso_he_objective(x, original_image):
    gamma = x[0]

    scaled_image = np.power(original_image / 255.0, gamma) * 255.0
    scaled_image = np.clip(scaled_image, 0, 255).astype(np.uint8)

    lower_thresh, upper_thresh = 50, 200
    _, thresholded_image = cv2.threshold(
        scaled_image, lower_thresh, 255, cv2.THRESH_TOZERO
    )
    thresholded_image = np.where(
        thresholded_image > upper_thresh, upper_thresh, thresholded_image
    ).astype(np.uint8)

    he_image = cv2.equalizeHist(thresholded_image)
    entropy = calculate_entropy(he_image)

    return -entropy


def apply_pso_he(original_image, lower_bound=0.95, upper_bound=1.05):
    initial_gamma = np.array([(lower_bound + upper_bound) / 2.0])

    result = minimize(
        pso_he_objective,
        initial_gamma,
        args=(original_image,),
        bounds=[(lower_bound, upper_bound)],
        method="L-BFGS-B"
    )

    optimal_gamma = result.x[0]

    scaled_image = np.power(original_image / 255.0, optimal_gamma) * 255.0
    scaled_image = np.clip(scaled_image, 0, 255).astype(np.uint8)

    lower_thresh, upper_thresh = 30, 150
    _, thresholded_image = cv2.threshold(
        scaled_image, lower_thresh, 255, cv2.THRESH_TOZERO
    )
    thresholded_image = np.where(
        thresholded_image > upper_thresh, upper_thresh, thresholded_image
    ).astype(np.uint8)

    optimized_he_image = cv2.equalizeHist(thresholded_image)

    return optimized_he_image, optimal_gamma
