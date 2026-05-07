import cv2
import numpy as np
from scipy.optimize import minimize
from preprocessing.pso_he import apply_pso_he


def apply_original(image):
    return image.astype(np.uint8)


def apply_he(image):
    image = image.astype(np.uint8)
    return cv2.equalizeHist(image)


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    image = image.astype(np.uint8)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    return clahe.apply(image)


def apply_clahe_he(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe_img = apply_clahe(image, clip_limit, tile_grid_size)
    return cv2.equalizeHist(clahe_img)


def pso_ie_objective(x, original_image):
    gamma = x[0]
    image = original_image.astype(np.float32) / 255.0
    enhanced = np.power(image, gamma)
    return -np.std(enhanced)


def apply_pso_ie(image, lower_bound=0.1, upper_bound=0.5):
    image = image.astype(np.uint8)
    initial_gamma = np.array([(lower_bound + upper_bound) / 2.0])

    result = minimize(
        pso_ie_objective,
        initial_gamma,
        args=(image,),
        bounds=[(lower_bound, upper_bound)],
        method="L-BFGS-B"
    )

    gamma = result.x[0]
    enhanced = np.power(image.astype(np.float32) / 255.0, gamma) * 255.0
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

    return enhanced


def apply_preprocessing(image, method="pso_he"):
    method = method.lower()

    if method in ["original", "none", "raw"]:
        return apply_original(image)

    if method == "he":
        return apply_he(image)

    if method == "clahe":
        return apply_clahe(image)

    if method in ["clahe_he", "clahe-he"]:
        return apply_clahe_he(image)

    if method in ["pso_ie", "pso-ie"]:
        return apply_pso_ie(image)

    if method in ["pso_he", "pso-he"]:
        enhanced, _ = apply_pso_he(image)
        return enhanced

    raise ValueError(
        f"Unknown preprocessing method: {method}. "
        "Choose from original, he, clahe, clahe_he, pso_ie, pso_he."
    )
