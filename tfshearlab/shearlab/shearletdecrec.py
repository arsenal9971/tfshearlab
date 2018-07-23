from numpy import ceil
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import numpy as np

__all__ = ('sheardec2D', 'shearrec2D', 'sheardecadjoint2D', 'shearrecadjoint2D')

def sheardec2D(X, shearletsystem):
    """Shearlet Decomposition function."""
    coeffs = np.zeros(shearletsystem.shearlets.shape, dtype=complex)
    Xfreq = fftshift(fft2(ifftshift(X)))
    for i in range(shearletsystem.nShearlets):
        coeffs[:, :, i] = fftshift(ifft2(ifftshift(Xfreq * np.conj(
                                   shearletsystem.shearlets[:, :, i]))))
    return coeffs.real


def shearrec2D(coeffs, shearletsystem):
    """Shearlet Recovery function."""
    X = np.zeros(coeffs.shape[:2], dtype=complex)
    for i in range(shearletsystem.nShearlets):
        X = X + fftshift(fft2(
            ifftshift(coeffs[:, :, i]))) * shearletsystem.shearlets[:, :, i]
    return (fftshift(ifft2(ifftshift((
            X / shearletsystem.dualFrameWeights))))).real


def sheardecadjoint2D(coeffs, shearletsystem):
    """Shearlet Decomposition adjoint function."""
    X = np.zeros(coeffs.shape[:2], dtype=complex)
    for i in range(shearletsystem.nShearlets):
        X = X + fftshift(fft2(
            ifftshift(coeffs[:, :, i]))) * np.conj(
            shearletsystem.shearlets[:, :, i])
    return (fftshift(ifft2(ifftshift(
            X / shearletsystem.dualFrameWeights)))).real


def shearrecadjoint2D(X, shearletsystem):
    """Shearlet Recovery adjoint function."""
    coeffs = np.zeros(shearletsystem.shearlets.shape, dtype=complex)
    Xfreq = fftshift(fft2(ifftshift(X)))
    for i in range(shearletsystem.nShearlets):
        coeffs[:, :, i] = fftshift(ifft2(ifftshift(
            Xfreq * shearletsystem.shearlets[:, :, i])))
    return coeffs.real

