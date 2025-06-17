import numpy as np

def generate_filament(shape, center, length, width, angle):
    """
    Generate a single filamentary (cylindrical) structure in 2D.

    Parameters:
    - shape: Tuple (nx, ny), the size of the 2D grid.
    - center: Tuple (x0, y0), the center of the filament.
    - length: Length of the filament.
    - width: Width (standard deviation) of the filament.
    - angle: Angle of the filament in degrees (0 = horizontal).

    Returns:
    - filament: 2D array containing the filamentary structure.
    """
    x = np.arange(0, shape[0])
    y = np.arange(0, shape[1])
    xx, yy = np.meshgrid(x, y)

    # Rotate the grid to align the filament with the given angle
    x_rot = (xx - center[0]) * np.cos(np.radians(angle)) + (yy - center[1]) * np.sin(np.radians(angle))
    y_rot = -(xx - center[0]) * np.sin(np.radians(angle)) + (yy - center[1]) * np.cos(np.radians(angle))

    # Create the filament as a Gaussian along the rotated x-axis
    filament = np.exp(-((x_rot**2) / (2 * (length / 2)**2) + (y_rot**2) / (2 * (width / 2)**2)))
    return filament

def generate_filament_field(size, a):
  """Generate a Gaussian random field with power law power spectrum.

  Args:
      size (int): the linear dimension of the field in pixels
      a (float): exponent of the power spectrum (P(k) ~ k^a)

  Returns:
      np.array: resulting scalar GRF of size (size, size)
      np.array: resulting eigenvalues of the CZ operator of size (size, size, 2)
  """
  # Create a grid of wave numbers
  kx = np.fft.fftfreq(size)
  ky = np.fft.fftfreq(size)
  kx, ky = np.meshgrid(kx, ky)
  k = np.sqrt(kx**2 + ky**2)

  # Generate random complex numbers with Gaussian distribution
  real_part = np.random.normal(size=(size, size))
  imag_part = np.random.normal(size=(size, size))
  random_field = real_part + 1j * imag_part

  # Apply the power spectrum
  power_spectrum = (k**a)
  power_spectrum[0, 0] = 0  # Avoid division by zero at the zero frequency
  field_ft = random_field * np.sqrt(power_spectrum)

  # apply the Caldéron-Zygmund operator
  field_ft_CZ = np.zeros((size, size,2,2), dtype=complex)
  field_ft_CZ[:,:,0,0] = (kx * kx)/k**2 * field_ft
  field_ft_CZ[:,:,0,1] = (kx * ky)/k**2 * field_ft
  field_ft_CZ[:,:,1,0] = field_ft_CZ[:,:,0,1]
  field_ft_CZ[:,:,1,1] = (ky * ky)/k**2 * field_ft

  field_ft_CZ[0,0,:,:] = 0
  # inverse Fourier transform
  field_CZ = np.fft.ifft2(field_ft_CZ,axes=(0,1)).real
  # compute eigenvalues
  field_CZ_eval = np.linalg.eigvalsh(field_CZ)

  # Perform the inverse FFT to get the spatial field
  field = np.fft.ifft2(field_ft).real

  norm = 1 / field.flatten().std()

  return field_CZ_eval * norm