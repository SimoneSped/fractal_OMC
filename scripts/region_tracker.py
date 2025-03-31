import scipy.ndimage as ndimage
from skimage.measure import perimeter
import numpy as np

def pca_major_axis(region_coords):
    """
    Computes the major axis length of a structure using PCA.
    The major axis is the square root of the largest eigenvalue of the covariance matrix.
    """
    if len(region_coords) < 2:
        return 0  # If only one pixel, size is zero

    # Center the coordinates
    centered_coords = region_coords - np.mean(region_coords, axis=0)

    # Compute covariance matrix and its eigenvalues
    cov_matrix = np.cov(centered_coords, rowvar=False)
    eigenvalues, _ = np.linalg.eigh(cov_matrix)

    # The largest eigenvalue corresponds to the major axis squared
    major_axis = 2 * np.sqrt(np.max(eigenvalues))  # Factor 2 to approximate full length

    return major_axis

def calculate_mass_and_size(region_mask, M_H2, pc_per_px):
    """
    Calculate the mass and the major axis of a structure using PCA.
    - Mass: Summed pixel values converted to solar masses.
    - Size: Major axis computed from PCA eigenvalues.
    """
    # Compute total mass
    mass = np.sum(M_H2[region_mask])

    # Find structure coordinates
    region_coords = np.column_stack(np.where(region_mask))

    if len(region_coords) == 0:
        return mass, 0  # If no region is found, return zero size

    # Compute the major axis length using PCA
    major_axis_length = pca_major_axis(region_coords)

    # Convert size to parsecs
    size_parsecs = major_axis_length * pc_per_px

    return mass, size_parsecs

def compute_fractal_dimension(region_mask):
    """
    Compute the fractal dimension of a binary region using the Perimeter-Area relation.
    
    Parameters:
        region_mask (ndarray): Boolean mask of the structure.
        
    Returns:
        float: Estimated fractal dimension.
    """
    area = np.sum(region_mask)
    
    # Compute perimeter using dilation technique
    mask_perimeter = perimeter(region_mask)

    if area > 0 and mask_perimeter > 0:
        return 2 * (np.log10(mask_perimeter) / np.log10(area))
    return None  # Invalid case (avoid division errors)

def track_largest_regions(N_H2, M_H2, thresholds, pc_per_px, num_top_regions=5):
    """
    Track the n largest regions across multiple thresholds and compute their fractal dimensions.
    
    Parameters:
        N_H2_OA (ndarray): 2D array of the column density map.
        thresholds (list): List of threshold values.
        num_top_regions (int): Number of largest regions to track.
    
    Returns:
        dict: Dictionary mapping the largest regions to their evolution history.
    """
    regions = {}

    for threshold in thresholds:
        mask = N_H2 >= threshold  
        labeled_regions, num_features = ndimage.label(mask)  # Identify structures
        
        if num_features == 0:
            continue  # No structures found at this threshold

        # Compute the size of each region
        region_sizes = {region_id: np.sum(labeled_regions == region_id) for region_id in range(1, num_features + 1)}

        # Get the top N largest regions
        top_regions = sorted(region_sizes.keys(), key=lambda k: region_sizes[k], reverse=True)[:num_top_regions]

        for region_id in top_regions:
            region_mask = labeled_regions == region_id
            fractal_dim = compute_fractal_dimension(region_mask)
            mass, size = calculate_mass_and_size(region_mask, M_H2, pc_per_px)

            # Store the evolution of each tracked region
            if region_id not in regions:
                regions[region_id] = {
                    "thresholds": [], 
                    "fractal_dimensions": [],
                    "mass": [],
                    "size": []
                }

            regions[region_id]["thresholds"].append(threshold)
            regions[region_id]["fractal_dimensions"].append(fractal_dim)
            regions[region_id]["mass"].append(mass)
            regions[region_id]["size"].append(size)

    return regions