import numpy as np

def generate_sierpinski_triangle(size, iterations):
    """
    Generate a Sierpinski Triangle using recursion.

    Parameters:
    - size: The size of the grid (size x size).
    - iterations: The number of iterations to perform.

    Returns:
    - triangle: A 2D numpy array representing the Sierpinski Triangle.
    """
    # Initialize the grid
    triangle = np.zeros((size, size), dtype=bool)

    # Define the initial triangle (equilateral)
    def draw_triangle(x, y, side_length):
        for i in range(side_length):
            for j in range(i + 1):
                triangle[x + i, y - j] = True
                triangle[x + i, y + j] = True

    # Recursive function to draw the Sierpinski Triangle
    def sierpinski(x, y, side_length, depth):
        if depth == 0:
            draw_triangle(x, y, side_length)
        else:
            half = side_length // 2
            sierpinski(x, y, half, depth - 1)
            sierpinski(x + half, y - half, half, depth - 1)
            sierpinski(x + half, y + half, half, depth - 1)

    # Start the recursion
    sierpinski(0, size // 2, size // 2, iterations)
    return triangle