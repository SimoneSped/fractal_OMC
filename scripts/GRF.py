import numpy as np
from scipy.signal import correlate2d
from scipy.optimize import curve_fit

## FUNCTION GENERATE A GAUSSIAN RANDOM FIELD

def generate_gaussian_random_field(size, a):
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

  return field * norm, field_CZ_eval * norm

def estimate_correlation_length(field, size):
    acf = correlate2d(field, field, mode="full")
    acf /= np.max(acf)  # Normalize

    # Step 3: Extract the 1D radial correlation function
    center = np.array(acf.shape) // 2
    radial_distances = np.arange(0, size//2)
    radial_acf = np.array([acf[center[0] + r, center[1]] for r in radial_distances])

    # Step 4: Fit an exponential decay function to estimate correlation length
    def exp_decay(r, xi):
        return np.exp(-r / xi)

    popt, _ = curve_fit(exp_decay, radial_distances, radial_acf, p0=[10])
    estimated_corr_length = popt[0]

    return estimated_corr_length
