import numpy as np

def box_counting(img):
    """
    Calculate the fractal dimension of a 2D binary image using the box-counting method.

    Parameters:
        img (numpy.ndarray): A 2D binary image where 1 represents the object and 0 represents the background.
        visualize (bool): If True, shows how the image is divided into boxes at each scale.

    Returns:
        float: The fractal dimension of the image.
    """
    # Sizes of boxes to use (powers of 2)
    sizes = 2 ** np.arange(1, int(np.log2(min(img.shape))) + 1)
    
    # Count the number of boxes that contain at least one pixel of the object
    box_counts = []
    
    for size in sizes:
        # Count the number of boxes that contain at least one pixel of the object
        box_count = 0
        for i in range(0, img.shape[0], size):
            for j in range(0, img.shape[1], size):
                if np.any(img[i:i+size, j:j+size]):
                    box_count += 1
        box_counts.append(box_count)

    # Log-log plot (log(sizes) vs log(box_counts))
    log_sizes = np.log(1/sizes)
    log_counts = np.log(box_counts)

    # Fit a line to find the slope (fractal dimension)
    slope, _ = np.polyfit(log_sizes, log_counts, 1)

    return slope