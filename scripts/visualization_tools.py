import matplotlib.pyplot as plt
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