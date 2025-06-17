import scipy.ndimage as ndimage
from skimage.measure import perimeter
import numpy as np

import pickle

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
        return 2*(np.log10(mask_perimeter) / np.log10(area))
    return None  # Invalid case (avoid division errors)

def find_YSOs_within_region(data_YSOs, region):
    pass

# import numpy as np
# from scipy import ndimage

# def track_largest_regions(N_H2, M_H2, thresholds, pc_per_px, num_top_regions=5, min_iou=0.3, min_pixels=10):
#     """
#     Track the n largest regions from the lowest threshold and follow their evolution through higher thresholds.

#     Parameters:
#         N_H2 (ndarray): 2D array of the column density map.
#         M_H2 (ndarray): 2D array of the mass map.
#         thresholds (list): List of threshold values, from low to high.
#         pc_per_px (float): Pixel scale in parsecs per pixel.
#         num_top_regions (int): Number of largest base regions to track.
#         min_iou (float): Minimum IoU to consider regions as matching across thresholds.
#         min_pixels (int): Minimum number of pixels to consider a region valid.

#     Returns:
#         dict: Dictionary mapping region IDs to a list of properties at each threshold.
#     """
#     assert np.all(np.diff(thresholds) >= 0), "Thresholds must be sorted from low to high."

#     base_threshold = thresholds[0]
#     base_mask = N_H2 >= base_threshold
#     labeled_base, num_base = ndimage.label(base_mask)

#     # Measure size and select top N regions
#     base_sizes = {
#         i: np.sum(labeled_base == i)
#         for i in range(1, num_base + 1)
#     }

#     top_region_ids = sorted(base_sizes, key=base_sizes.get, reverse=True)[:num_top_regions]
#     tracked_regions = {}

#     for region_id in top_region_ids:
#         region_mask = labeled_base == region_id
#         if np.count_nonzero(region_mask) < min_pixels:
#             continue
#         tracked_regions[region_id] = {
#             "base_mask": region_mask,
#             "threshold_data": []
#         }

#     # Track each region through increasing thresholds
#     for threshold in thresholds:
#         current_mask = N_H2 >= threshold
#         labeled, num_features = ndimage.label(current_mask)

#         for tracked_id, region_info in tracked_regions.items():
#             base_mask = region_info["base_mask"]
#             best_iou = 0
#             best_region = None

#             for i in range(1, num_features + 1):
#                 candidate_mask = labeled == i
#                 intersection = np.logical_and(base_mask, candidate_mask).sum()
#                 union = np.logical_or(base_mask, candidate_mask).sum()
#                 if union == 0:
#                     continue
#                 iou = intersection / union

#                 if iou > best_iou and iou >= min_iou:
#                     best_iou = iou
#                     best_region = candidate_mask

#             if best_region is not None and np.count_nonzero(best_region) >= min_pixels:
#                 fractal_dim = compute_fractal_dimension(best_region)
#                 mass, size = calculate_mass_and_size(best_region, M_H2, pc_per_px)

#                 if fractal_dim < 2.01:
#                     region_info["threshold_data"].append({
#                         "threshold": threshold,
#                         "fractal_dimension": fractal_dim,
#                         "mass": mass,
#                         "size": size,
#                         "region_mask": best_region.copy(),
#                         "YSOs": []  # Optional: fill in later
#                     })

#     return tracked_regions

def track_largest_regions(N_H2, M_H2, thresholds, pc_per_px, num_top_regions=5):
    """
    Track the n largest regions across multiple thresholds and compute their fractal dimensions.
    
    Parameters:
        N_H2 (ndarray): 2D array of the column density map.
        M_H2 (ndarray): 2D array of the mass map.
        thresholds (list): List of threshold values.
        pc_per_px (float): Pixel scale in parsecs per pixel.
        num_top_regions (int): Number of largest regions to track.
    
    Returns:
        dict: Dictionary mapping unique region identifiers to their properties.
    """
    regions = {}

    for threshold in thresholds:
        mask = N_H2 >= threshold  
        labeled_regions, num_features = ndimage.label(mask)
        
        if num_features == 0:
            continue

        region_sizes = {
            region_id: np.sum(labeled_regions == region_id)
            for region_id in range(1, num_features + 1)
        }

        top_regions = sorted(region_sizes.keys(), key=lambda k: region_sizes[k], reverse=True)[:num_top_regions]

        for region_id in top_regions:
            region_mask = labeled_regions == region_id
            fractal_dim = compute_fractal_dimension(region_mask)
            mass, size = calculate_mass_and_size(region_mask, M_H2, pc_per_px)
            # YSOs_list = find_YSOs_within_region()

            # Unique key per threshold+region
            region_key = f"{threshold}_{region_id}"
            regions[region_key] = {
                "threshold": threshold,
                "fractal_dimension": fractal_dim,
                "mass": mass,
                "size": size,
                "region_mask": region_mask,
                "YSOs": []  # Add real values later
            }

    return regions


def save_regions_to_pickle(regions, filename):
    with open(filename, 'wb') as f:
        pickle.dump(regions, f)

def load_regions_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_region_from_fractal_dimension_mass_and_size(regions, fractal_dim=None, size=None, mass=None, tolerance=0.1):
    """
    Extract regions that match the given fractal dimension, size, and/or mass criteria.

    Parameters:
        regions (dict): Dictionary of regions with their properties.
        fractal_dim (float, optional): Target fractal dimension to match.
        size (float, optional): Target size to match.
        mass (float, optional): Target mass to match.
        tolerance (float): Allowed relative tolerance for matching (default is 10%).

    Returns:
        list: List of region IDs that match the criteria.
    """
    matching_regions = []

    for region_id, properties in regions.items():
        matches = True

        if fractal_dim is not None:
            avg_fractal_dim = np.mean(properties["fractal_dimensions"])
            if not (avg_fractal_dim * (1 - tolerance) <= fractal_dim <= avg_fractal_dim * (1 + tolerance)):
                matches = False

        if size is not None:
            avg_size = np.mean(properties["size"])
            if not (avg_size * (1 - tolerance) <= size <= avg_size * (1 + tolerance)):
                matches = False

        if mass is not None:
            avg_mass = np.mean(properties["mass"])
            if not (avg_mass * (1 - tolerance) <= mass <= avg_mass * (1 + tolerance)):
                matches = False

        if matches:
            matching_regions.append(region_id)

    return matching_regions