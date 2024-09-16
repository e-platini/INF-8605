import cv2
import numpy as np
from scipy.ndimage import label, center_of_mass, gaussian_filter


def create_gaussian_blob(array, x, y, sigma=8):
    """ Draw a Gaussian blob centered at (x, y) on a proxy empty array and add this array to the input array """
    temp_array = np.zeros(array.shape[:2])
    temp_array[y, x] = 1  # Set the initial point
    gaussian_filter(temp_array, sigma=sigma, output=temp_array, mode='constant', cval=0)  # Filter the whole array
    array[:, :, 0] = np.maximum(array[:, :, 0], temp_array)  # Use maximum to avoid stacking intensities


def get_centroids(grayscale_mask):
    # INPUT SHOULD BE A grayscale MASK with values 0 AND 255 AND BE OF TYPE UINT8
    # _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)

    # Calculate the median value
    cuttof = np.percentile(grayscale_mask, 99)

    # Create a binary mask where values above the median are set to 255 and others to 0
    grayscale_mask = np.where(grayscale_mask > cuttof, 255, 0).astype(np.uint8)

    # Find sure background
    sure_bg = cv2.dilate(grayscale_mask, np.ones((3, 3), np.uint8), iterations=3)

    # Compute the distance transform for sure foreground
    dist_transform = cv2.distanceTransform(grayscale_mask, cv2.DIST_L2, 5)

    # Threshold to get the sure foreground
    _, sure_fg = cv2.threshold(dist_transform, 0.35 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Find unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(cv2.cvtColor(grayscale_mask, cv2.COLOR_GRAY2BGR), markers)
    grayscale_mask[markers == -1] = 0  # Boundary marked with -1

    # Calculate centroids using the markers
    centroids = []
    unique_markers = np.unique(markers)
    for marker in unique_markers:
        if marker > 1:  # Ignore the background and boundary markers
            mask = np.zeros_like(markers)
            mask[markers == marker] = 1
            centroid = center_of_mass(mask)
            centroids.append(centroid)

    # Create an empty 3D array
    output_array = np.zeros((grayscale_mask.shape[0], grayscale_mask.shape[1], 1), dtype=np.float32)
    for centroid in centroids:
        create_gaussian_blob(output_array, int(centroid[1]), int(centroid[0]))

    return output_array
