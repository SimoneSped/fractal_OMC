import numpy as np
import matplotlib.pyplot as plt

def sierpinski_triangle(n):
    """Generate a Sierpinski Triangle of order n in a 2D binary array."""
    size = 2 ** n
    triangle = np.zeros((size, size), dtype=int)

    # Function to fill in the Sierpinski Triangle pattern
    def fill_triangle(x, y, size):
        if size == 1:
            triangle[x, y] = 1
        else:
            new_size = size // 2
            fill_triangle(x, y, new_size)
            fill_triangle(x + new_size, y, new_size)
            fill_triangle(x, y + new_size, new_size)

    fill_triangle(0, 0, size)
    return triangle

def box_counting_fractal_dimension(mask):
    """Estimate the fractal dimension of a binary mask using the box-counting method."""
    binary_image = mask.astype(int)
    
    # Define a range of box sizes (logarithmically spaced)
    min_box_size = 2  # Minimum box size (should be >= 2)
    max_box_size = min(mask.shape) // 2  # Maximum box size
    box_sizes = np.logspace(np.log2(min_box_size), np.log2(max_box_size), num=10, base=2, dtype=int)

    counts = []
    for size in box_sizes:
        count = 0
        for x in range(0, mask.shape[0], size):
            for y in range(0, mask.shape[1], size):
                if np.any(binary_image[x:x+size, y:y+size]):
                    count += 1
        counts.append(count)
    
    # Perform a linear fit on the log-log plot
    log_box_sizes = np.log(1 / np.array(box_sizes))
    log_counts = np.log(counts)
    coeffs = np.polyfit(log_box_sizes, log_counts, 1)
    fractal_dimension = coeffs[0]

    # Plot the log-log relationship
    plt.figure(figsize=(8, 6))
    plt.plot(log_box_sizes, log_counts, 'o-', label="Box-Counting Data")
    plt.plot(log_box_sizes, np.polyval(coeffs, log_box_sizes), 'r--', label=f"Fit Line (Slope = {fractal_dimension:.2f})")
    plt.xlabel("log(1 / Box Size)")
    plt.ylabel("log(Number of Boxes)")
    plt.title("Fractal Dimension Estimate for Sierpinski Triangle (circa 1.585)")
    plt.legend()
    plt.show()

    return fractal_dimension

if __name__ == '__main__':
    # Generate and plot a Sierpinski Triangle of order 6
    sierpinski = sierpinski_triangle(6)
    plt.imshow(sierpinski, cmap='gray', origin='lower')
    plt.title("Sierpinski Triangle (Order 6)")
    plt.show()

    # Estimate the fractal dimension of the Sierpinski Triangle
    true_value = np.log(3)/np.log(2)
    fractal_dim = box_counting_fractal_dimension(sierpinski)
    print(f"Estimated Fractal Dimension of the Sierpinski Triangle: {fractal_dim:.3f}")
    print("Absolute Error", true_value-fractal_dim)
    print("Relative Error [%]", 100*np.abs((true_value-fractal_dim)/true_value))