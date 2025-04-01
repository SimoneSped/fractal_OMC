import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

import porespy as ps

# Visualization Style
ps.visualization.set_mpl_style()

def show_regions_with_sections(data, n_regions=3, thresholds = None):
    # Define the color limits based on percentiles
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    min_value = np.percentile(data, 2)  # 2nd percentile
    max_value = np.percentile(data, 98)  # 98th percentile

    if n_regions == 0:
        pass
    else:
        # Plot the entire data map
        plt.figure(figsize=(10, 6))
        plt.imshow(data, vmin=min_value, vmax=max_value, interpolation=None, origin='lower', cmap='inferno')

        height, width = data.shape
        # Determine step size for slicing along the y-axis
        w_step = width // n_regions

        # Draw vertical lines to indicate section boundaries and add region labels
        for i in range(1, n_regions):
            plt.axvline(x=i * w_step, color="red", linestyle="--", linewidth=1.5)
            plt.text(i * w_step - w_step // 2, height // 10, f'Region {i}', color='white', fontsize=12, ha='center', va='top')

        # Add label for the last region
        plt.text(width - w_step // 2, height // 10, f'Region {n_regions}', color='white', fontsize=12, ha='center', va='top')

        plt.title("Map with Section Boundaries")
        plt.colorbar(label="Value")
        plt.tight_layout()
        plt.show()

    if thresholds:
        for threshold in thresholds:
            # Create mask for the given threshold
            mask = data >= threshold
            masked_data = np.where(mask, data, np.nan)

            # Define the color limits based on percentiles
            min_value = np.percentile(masked_data[np.isfinite(masked_data)], 2)  # 2nd percentile
            max_value = np.percentile(masked_data[np.isfinite(masked_data)], 98)  # 98th percentile

            # Plot the mask
            plt.figure(figsize=(10, 8))
            plt.imshow(masked_data, vmin=min_value, vmax=max_value, origin='lower', cmap='inferno', interpolation=None)

            if n_regions != 0:
                # Draw vertical lines to indicate section boundaries and add region labels
                for i in range(1, n_regions):
                    plt.axvline(x=i * w_step, color="red", linestyle="--", linewidth=1.5)
                    plt.text(i * w_step - w_step // 2, height // 10, f'Region {i}', color='black', fontsize=12, ha='center', va='top')

                # Add label for the last region
                plt.text(width - w_step // 2, height // 10, f'Region {n_regions}', color='black', fontsize=12, ha='center', va='top')

            plt.title(f"Map with Threshold {threshold:.2e}")
            plt.colorbar(label="Value")
            plt.tight_layout()
            plt.show()

def show_Minkowski_Functionals(region_name, results):
    plt.figure(figsize=(14, 8))
    plt.suptitle("Minkowski Functionals of "+region_name, fontsize=16)

    plt.subplot(1, 4, 1)
    plt.plot(results["thresholds"], results["areas"], 'o-')
    plt.xlabel('Threshold')
    plt.ylabel('Area (v0)')

    plt.subplot(1, 4, 2)
    plt.plot(results["thresholds"], results["perimeters"], 'o-')
    plt.xlabel('Threshold')
    plt.ylabel('Perimeter (v1)')

    plt.subplot(1, 4, 3)
    plt.plot(np.log10(results["perimeters"]), np.log10(results["areas"]), 'o-')
    plt.xlabel('log P')
    plt.ylabel('log A')

    plt.subplot(1, 4, 4)
    plt.plot(results["thresholds"], results["euler_chars"], 'o-', label='Euler Characteristic')
    plt.xlabel('Threshold')
    plt.ylabel('Euler Characteristic (v2)')

    plt.tight_layout()
    plt.show()

def show_sections_OrionB_vertical(data, n_regions=3, thresholds=None):
    """
    Really specific case for OrionB data, where we want to show the sections vertically.
    """
    height, width = data.shape
    # Determine step size for slicing along the y-axis
    h_step = height // n_regions

    # Define the color limits based on percentiles
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    min_value = np.percentile(data, 2)  # 2nd percentile
    max_value = np.percentile(data, 98)  # 98th percentile

    # Plot the entire data map
    plt.figure(figsize=(10, 6))
    plt.imshow(data, vmin=min_value, vmax=max_value, interpolation=None, origin='lower', cmap='inferno')

    # Draw horizontal lines to indicate section boundaries and add region labels
    for i in range(1, n_regions):
        plt.axhline(y=i * h_step, color="red", linestyle="--", linewidth=1.5)
        plt.text(width // 10, i * h_step - h_step // 2, f'Region {i}', color='white', fontsize=12, ha='center', va='top')

    # Add label for the last region
    plt.text(width // 10, height - h_step // 2, f'Region {n_regions}', color='white', fontsize=12, ha='center', va='top')

    plt.title("Map with Section Boundaries")
    plt.colorbar(label="Value")
    plt.tight_layout()
    plt.show()

    if thresholds:
        for threshold in thresholds:
            # Create mask for the given threshold
            mask = data >= threshold
            masked_data = np.where(mask, data, np.nan)

            # Define the color limits based on percentiles
            min_value = np.percentile(masked_data[np.isfinite(masked_data)], 2)  # 2nd percentile
            max_value = np.percentile(masked_data[np.isfinite(masked_data)], 98)  # 98th percentile

            # Plot the mask
            plt.figure(figsize=(10, 8))
            plt.imshow(masked_data, vmin=min_value, vmax=max_value, origin='lower', cmap='inferno', interpolation=None)

            # Draw horizontal lines to indicate section boundaries and add region labels
            for i in range(1, n_regions):
                plt.axhline(y=i * h_step, color="red", linestyle="--", linewidth=1.5)
                plt.text(width // 10, i * h_step - h_step // 2, f'Region {i}', color='black', fontsize=12, ha='center', va='top')

            # Add label for the last region
            plt.text(width // 10, height - h_step // 2, f'Region {n_regions}', color='black', fontsize=12, ha='center', va='top')

            plt.title(f"Map with Threshold {threshold:.2e}")
            plt.colorbar(label="Value")
            plt.tight_layout()
            plt.show()


def visualize_Mass_Size_D_Diagrams(fractal_dims_OA, fractal_dims_OB, masses_OA, masses_OB, sizes_OA, sizes_OB):

    # Determine the global min and max fractal dimension
    global_vmin = min(np.min(fractal_dims_OA), np.min(fractal_dims_OB))
    global_vmax = max(np.max(fractal_dims_OA), np.max(fractal_dims_OB))

    # Create a common normalization for both datasets
    norm = mcolors.Normalize(vmin=global_vmin, vmax=global_vmax)
    cmap = plt.cm.viridis  # Choose your preferred colormap

    # Plot the lines and data points
    plt.figure(figsize=(10,6))

    # Define the extended mass range
    M_min = np.min(masses_OA) if np.min(masses_OA) > 0 else 1  # Avoid zero or negative values
    M_max = np.max(masses_OA)  # Extend lines to the highest observed mass
    M = np.logspace(np.log10(M_min), np.log10(M_max), 100)

    # Constants
    b0 = 1e4  # cm^(-2)
    G = 4.3e-3  # pc M_sun^-1 (km/s)^2
    sigma = 2  # Assumed velocity dispersion in km/s (adjust if needed)

    # Aspect Ratio 10
    A_10 = 10
    a_10 = (2 * A_10) / (np.pi * b0) ** 0.5
    L_10 = a_10 * M ** 0.5

    # Aspect Ratio 3
    A_3 = 3
    a_3 = (2 * A_3) / (np.pi * b0) ** 0.5
    L_3 = a_3 * M ** 0.5

    # Virial mass line (M_vir)
    L_vir = (G * M) / (5 * sigma**2)  # Using M_vir = (5 sigma^2 L) / G

    plt.plot(M, L_10, label="Aspect Ratio=  10", linestyle="--", color="blue")
    plt.plot(M, L_3, label="Aspect Ratio = 3", linestyle="--", color="red")
    # plt.plot(M, L_vir, label="Virial Mass", linestyle="--", color="green")

    # First dataset
    sc1 = plt.scatter(masses_OA, sizes_OA, c=fractal_dims_OA, cmap=cmap, norm=norm, edgecolor='k', alpha=0.7)
    plt.colorbar(sc1, label='Fractal Dimension (Common Scale)')
    plt.xlabel('Mass (Solar Masses)')
    plt.ylabel('Size (pc)')
    plt.xscale("log")
    plt.yscale("log")
    plt.title('Mass vs. Size of Regions in Orion A with Fractal Dimension as Color')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()

    # Second dataset
    plt.figure(figsize=(10, 6))

    # Define the extended mass range
    M_min = np.min(masses_OB) if np.min(masses_OB) > 0 else 1  # Avoid zero or negative values
    M_max = np.max(masses_OB)  # Extend lines to the highest observed mass
    M = np.logspace(np.log10(M_min), np.log10(M_max), 100)

    # Aspect Ratio 10
    A_10 = 10
    a_10 = (2 * A_10) / (np.pi * b0) ** 0.5
    L_10 = a_10 * M ** 0.5

    # Aspect Ratio 3
    A_3 = 3
    a_3 = (2 * A_3) / (np.pi * b0) ** 0.5
    L_3 = a_3 * M ** 0.5

    L_vir = (G * M) / (5 * sigma**2)

    plt.plot(M, L_10, label="Aspect Ratio=  10", linestyle="--", color="blue")
    plt.plot(M, L_3, label="Aspect Ratio = 3", linestyle="--", color="red")
    # plt.plot(M, L_vir, label="Virial Mass", linestyle="--", color="green")

    sc2 = plt.scatter(masses_OB, sizes_OB, c=fractal_dims_OB, cmap=cmap, norm=norm, edgecolor='k', alpha=0.7)
    plt.colorbar(sc2, label='Fractal Dimension (Common Scale)')
    plt.xlabel('Mass (Solar Masses)')
    plt.ylabel('Size (pc)')
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.title('Mass vs. Size of Regions in Orion B with Fractal Dimension as Color')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

def visualize_Mass_Size_D_Diagramm_as_One(fractal_dims_OA, fractal_dims_OB, masses_OA, masses_OB, sizes_OA, sizes_OB):
    # Determine the global min and max fractal dimension
    global_vmin = min(np.min(fractal_dims_OA), np.min(fractal_dims_OB))
    global_vmax = max(np.max(fractal_dims_OA), np.max(fractal_dims_OB))

    # Create a common normalization for both datasets
    norm = mcolors.Normalize(vmin=global_vmin, vmax=global_vmax)
    cmap = plt.cm.viridis  # Choose your preferred colormap

    # Plot the lines and data points
    plt.figure(figsize=(10, 6))

    # Define the extended mass range
    M_min = min(np.min(masses_OA), np.min(masses_OB)) if np.min(masses_OA) > 0 and np.min(masses_OB) > 0 else 1  # Avoid zero or negative values
    M_max = max(np.max(masses_OA), np.max(masses_OB))  # Extend lines to the highest observed mass
    M = np.logspace(np.log10(M_min), np.log10(M_max), 100)

    # Constants
    b0 = 1e4  # cm^(-2)
    G = 4.3e-3  # pc M_sun^-1 (km/s)^2
    sigma = 2  # Assumed velocity dispersion in km/s (adjust if needed)

    # Aspect Ratio 10
    A_10 = 10
    a_10 = (2 * A_10) / (np.pi * b0) ** 0.5
    L_10 = a_10 * M ** 0.5

    # Aspect Ratio 3
    A_3 = 3
    a_3 = (2 * A_3) / (np.pi * b0) ** 0.5
    L_3 = a_3 * M ** 0.5

    # Virial mass line (M_vir)
    L_vir = (G * M) / (5 * sigma**2)  # Using M_vir = (5 sigma^2 L) / G

    # Plot the virial mass lines
    plt.plot(M, L_10, label="Aspect Ratio=  10", linestyle="--", color="blue")
    plt.plot(M, L_3, label="Aspect Ratio = 3", linestyle="--", color="red")
    # plt.plot(M, L_vir, label="Virial Mass", linestyle="--", color="green")

    # Plot data for Orion A
    sc1 = plt.scatter(masses_OA, sizes_OA, c=fractal_dims_OA, cmap=cmap, norm=norm, edgecolor='k', alpha=0.7)
    # Plot data for Orion B
    sc2 = plt.scatter(masses_OB, sizes_OB, c=fractal_dims_OB, cmap=cmap, norm=norm, edgecolor='k', alpha=0.7)

    # Add colorbars
    plt.colorbar(sc1, label='Fractal Dimension (Common Scale)')

    # Set labels and title
    plt.xlabel('Mass (Solar Masses)')
    plt.ylabel('Size (pc)')
    plt.xscale("log")
    plt.yscale("log")
    plt.title('Mass vs. Size of Regions in Orion A and B with Fractal Dimension as Color')

    # Add grid and legend
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(loc='upper left')

    # Show plot
    plt.show()


# For Simulations
def show_shapes_results(shape_image, name, thresholds, results, fractal_dimension_BC):
    # Display the line image
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(shape_image, cmap='gray')
    plt.title(name)
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.plot(thresholds, ((results["fractal_dimension"])), "o-")
    plt.xlabel('Threshold')
    plt.ylabel('D (Minkowski Functionals)')
    plt.title('Minkowski Functionals for '+name)

    # Calculate fractal dimensions using box counting method

    plt.subplot(1, 3, 3)
    plt.plot(thresholds, fractal_dimension_BC, '-o')
    plt.xlabel('Threshold')
    plt.ylabel('D (Box Counting)')
    plt.title('Fractal Dimensions for '+name)

    plt.tight_layout()
    plt.show()

def plot_GRF(field, field_CZ_eval, size):
  # Display the GRF and Eigenvalues 1 and 2 in a (1, 3) subplot
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(field, cmap='inferno', origin='lower', extent=[0, size, 0, size])
    plt.colorbar(label="Field Intensity")
    plt.title("Gaussian Random Field")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.subplot(1, 3, 2)
    plt.imshow(field_CZ_eval[:, :, 0], cmap='coolwarm', origin='lower', extent=[0, size, 0, size])
    plt.colorbar(label="Eigenvalue 1")
    plt.title("CZ Eigenvalue 1")

    plt.subplot(1, 3, 3)
    plt.imshow(field_CZ_eval[:, :, 1], cmap='coolwarm', origin='lower', extent=[0, size, 0, size])
    plt.colorbar(label="Eigenvalue 2")
    plt.title("CZ Eigenvalue 2")

    plt.tight_layout()
    plt.show()