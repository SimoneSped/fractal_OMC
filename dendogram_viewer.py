from astrodendro import Dendrogram
from astrodendro.analysis import PPStatistic
import logging as log
# For plots
import matplotlib.pyplot as plt

from astropy.wcs import WCS

import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u

import os
import sys
from astropy.io import fits
import h5py

def read_fits(filepath, print_header):
    # Check if the FITS file exists
    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}")
    else:
        try:
            # Open the FITS file
            with fits.open(filepath) as hdul:
                # Print a summary of the HDUs (Header Data Units)
                hdul.info()

                # Access the primary HDU data (usually image data)
                primary_hdu = hdul[0].data
                
                # Access the header of the primary HDU
                header = hdul[0].header
                
                if print_header:
                    print("Header Information (without blank spaces):")
                    print("-" * 60)  # Separator line
                    for key in header.keys():
                        value = header[key]
                        comment = header.comments[key]
                        # Remove extra spaces from key, value, and comment
                        clean_key = key.strip()
                        clean_value = str(value).strip()
                        clean_comment = comment.strip()
                        print(f"{clean_key}={clean_value}#{clean_comment}")  # Format output without spaces

        except Exception as e:
            print(f"An error occurred while opening the FITS file: {e}")
    
    return primary_hdu, header

def create_dendograms():

    # Get the current working directory
    curr_folder = os.getcwd()

    # Find the index of 'notebooks' in the current path
    notebooks_index = curr_folder.rfind('notebooks')

    # Check if 'notebooks' is found in the path
    if notebooks_index != -1:
        # Set the directory to the parent of 'notebooks'
        src_path = os.path.dirname(curr_folder[:notebooks_index])
        os.chdir(src_path)  # Change the current working directory to the source path
        sys.path.insert(0, src_path)  # Insert the source path into sys.path for module imports
        
    # Lombardi et al
    # the catalogue name in VizieR
    CATALOGUE_LOMBARDI = "J/A+A/566/A45"

    CATALOGUE_MEGEATH = "J/AJ/144/192"

    # Construct the path to the FITS file
    planck_herschel_fits_file = os.path.join(curr_folder, "Lombardi", "planck_herschel.fits.gz")

    hdu_herschel_fits_data, hdu_herschel_fits_header = read_fits(planck_herschel_fits_file, print_header=False)

    # Constructt WCS and image data
    wcs = WCS(hdu_herschel_fits_header)

    image_data = hdu_herschel_fits_data[0]

    # Replace NaNs and Infs with some valid value, e.g., zero or the median
    tau = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 1: Define constants
    # TO-DO: differentiate between Orion A and B 

    # Orion A: 206 ≤ l ≤ 217, −21 ≤ b ≤ −17
    # Orion B: 203 ≤ l ≤ 210, −17 ≤ b ≤ −12

    # Step 3: Apply the formula to calculate A_k (Lomabrdi et al)
    l_min_A, l_max_A = 206, 217
    b_min_A, b_max_A = -21, -17

    # Making it smaller cause of computation time
    l_min_A, l_max_A = 210, 212
    b_min_A, b_max_A = -21, -20

    min_coord_A = SkyCoord(l_min_A, b_min_A, frame='galactic', unit=u.deg)
    max_coord_A = SkyCoord(l_max_A, b_max_A, frame='galactic', unit=u.deg)

    min_pixel_A = wcs[:][:][0].world_to_pixel(min_coord_A)
    max_pixel_A = wcs[:][:][0].world_to_pixel(max_coord_A)

    gamma_orion_A = 2640  # mag
    delta_orion_A = 0.012  # mag, Offset for Orion A

    A_k = gamma_orion_A * tau + delta_orion_A

    A_k[int(min_pixel_A[1]): int(max_pixel_A[1]), int(max_pixel_A[0]): int(min_pixel_A[0])] = gamma_orion_A * tau[int(min_pixel_A[1]): int(max_pixel_A[1]), int(max_pixel_A[0]): int(min_pixel_A[0])] + delta_orion_A

    l_min_B, l_max_B = 203, 210
    b_min_B, b_max_B = -17, -12

    min_coord_B = SkyCoord(l_min_B, b_min_B, frame='galactic', unit=u.deg)
    max_coord_B = SkyCoord(l_max_B, b_max_B, frame='galactic', unit=u.deg)

    min_pixel_B = wcs[:][:][0].world_to_pixel(min_coord_B)
    max_pixel_B = wcs[:][:][0].world_to_pixel(max_coord_B)

    gamma_orion_B = 3460  # mag
    delta_orion_B = -0.001  # mag, Offset for Orion B

    A_k[int(min_pixel_B[1]): int(max_pixel_B[1]), int(max_pixel_B[0]): int(min_pixel_B[0])] = gamma_orion_B * tau[int(min_pixel_B[1]): int(max_pixel_B[1]), int(max_pixel_B[0]): int(min_pixel_B[0])] + delta_orion_B

    # A_k to A_V
    A_V = A_k/0.112

    # N(H2)
    N_H2 = 0.93e21 * np.array(A_V , dtype=np.float64)

    """
    This if mass is needed
    pixel_scale = 0.00417
    distance = 420
    radians = 180/np.pi #conversion factor: rad in deg
    rad_per_px = pixel_scale/radians
    pc_per_px = np.sin(rad_per_px)*distance
    pc2_per_px = pc_per_px**2
    cm_per_pc = 3.086*10**18
    cm2_per_px = pc2_per_px * (cm_per_pc ** 2) 

    m_p = 1.67e-27 # mass of proton (kg)

    M_H2 = np.array(N_H2, dtype=np.float64)*2.8*m_p/(1.98e30)

    # M_H2 = M_H2*area # check here!
    M_H2 = M_H2 * cm2_per_px

    # Assuming M_H2_clean is already defined
    M_H2_clean = np.nan_to_num(M_H2, nan=0.0, posinf=0.0, neginf=0.0)

    # Masking out negative values
    M_H2_clean_positive = np.where(M_H2_clean > 0, M_H2_clean, 0)

    M_H2_OA = M_H2_clean_positive[int(min_pixel_A[1]): int(max_pixel_A[1]), int(max_pixel_A[0]): int(min_pixel_A[0])]
    M_H2_OB = M_H2_clean_positive[int(min_pixel_B[1]): int(max_pixel_B[1]), int(max_pixel_B[0]): int(min_pixel_B[0])]
    """

    N_H2_OA = N_H2[int(min_pixel_A[1]): int(max_pixel_A[1]), int(max_pixel_A[0]): int(min_pixel_A[0])]
    N_H2_OB = N_H2[int(min_pixel_A[1]): int(max_pixel_A[1]), int(max_pixel_A[0]): int(min_pixel_A[0])]


    # Set the path to the dendrogram file
    dendrogram_filepath = os.path.join(os.getcwd(), "dendrogram_a.hdf5")

    # Check if the file exists
    if os.path.exists(dendrogram_filepath):
        print("No Dendogram found, computing!")
        # Compute and save the dendrogram if the file doesn't exist
        dendrogram_OA = Dendrogram.compute(N_H2_OA)
        dendrogram_OA.save_to(dendrogram_filepath)
    else:
        # TO-DO
        # Load the dendrogram from the file if it exists
        # dendrogram_OA = Dendrogram.load_from(dendrogram_filepath)
        log.debug('Loading HDF5 file from disk...')
        with h5py.File(dendrogram_filepath, 'r') as h5f:
            newick = h5f['newick'][()]
            if not isinstance(newick, str):
                    newick = newick.decode()
            dendrogram_OA = h5f['data'][()]
            dendrogram_OA = h5f['index_map'][()]

    v = dendrogram_OA.viewer()
    v.show()

    """
    # do this with 
    # obtain dendogram slices

    # do fractal dimension stuff with them

    def box_counting_fractal_dimension(mask):
        # Convert mask to binary image
        binary_image = mask.astype(int)
        
        # Define a range of box sizes
        box_sizes = np.logspace(1, np.log2(min(mask.shape)), num=10, base=2, dtype=int)

        counts = []
        for size in box_sizes:
            # Divide the image into boxes of the given size and count non-empty boxes
            count = 0
            for x in range(0, mask.shape[0], size):
                for y in range(0, mask.shape[1], size):
                    if np.any(binary_image[x:x+size, y:y+size]):
                        count += 1
            counts.append(count)
        # print(counts)

        # Fit a line to log(counts) vs log(1/box_size) to estimate the fractal dimension
        log_box_sizes = np.log(1 / np.array(box_sizes))
        log_counts = np.log(counts)
        
        # Perform a linear fit
        coeffs = np.polyfit(log_box_sizes, log_counts, 1)
        fractal_dimension = coeffs[0]
        return fractal_dimension

    # Example usage for one structure
    fractal_dimensions = []
    for leaf in dendrogram_OA.leaves:
        mask = leaf.get_mask()
        
        
 
        # Assuming 2D data for visualization
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original data plot
        ax[0].imshow(N_H2_OA, origin='lower', cmap='gray')
        ax[0].contour(mask, colors='red', linewidths=1)  # Overlay mask borders in red
        ax[0].set_title(f"Original Data with Mask Overlay (Leaf {leaf.idx})")
        
        # Mask plot
        ax[1].imshow(mask, origin='lower', cmap='gray')
        ax[1].set_title(f"Mask of Leaf {leaf.idx}")

            
        plt.show()
        
        fractal_dim = box_counting_fractal_dimension(mask)
        fractal_dimensions.append([leaf.idx, fractal_dim])
        print(f"Fractal dimension for structure {leaf.idx} out of {len(dendrogram_OA.leaves)}: {fractal_dim}")

    fractal_dimensions.sort(key=lambda x: x[0])

    # Extract indices and fractal dimensions for plotting
    indices = [fd[0] for fd in fractal_dimensions]
    dimensions = [fd[1] for fd in fractal_dimensions]

    # Plot the fractal dimension evolution
    plt.figure(figsize=(10, 6))
    plt.plot(indices, dimensions, marker='o', linestyle='-')
    plt.xlabel("Leaf Index (Hierarchy Level)")
    plt.ylabel("Fractal Dimension")
    plt.title("Fractal Dimension Evolution from Lower to Higher Leaves")
    plt.grid(visible=True, linestyle='--', alpha=0.5)
    plt.show()
    """

if __name__ == '__main__':
    create_dendograms()