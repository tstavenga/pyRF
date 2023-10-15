from typing import Any
import numpy as np
import scipy


class Eigenfunction:
    def __init__(self, 
                 coefficients: np.ndarray, 
                 channel_limits: dict, 
                 resonance_frequency_k,
                 min_position: float,
                 max_position: float) -> None:
        
        self.coefficients = coefficients
        self.channel_limits = channel_limits
        self.resonance_frequency_k = resonance_frequency_k
        self.min_position = min_position
        self.max_position = max_position

        self.channel_eigenfunctions = self.get_channel_eigenfunctions()
        normalization_factor = self.normalization_factor()

        # normalize the eigenfunction coefficients
        for channel in self.coefficients.keys():
            self.coefficients[channel] /= normalization_factor

        # update the eigenfunctions with the new normalization factor
        self.channel_eigenfunctions = self.get_channel_eigenfunctions()
            
    def get_channel_eigenfunctions(self):
        channel_eigenfunctions = []
        for channel in self.channel_limits.keys():        
            channel_eigenfunctions.append(lambda z,channel=channel: np.dot(self.coefficients[channel], 
                                                                                self.basis(self.resonance_frequency_k, z)))
        return channel_eigenfunctions

    def __call__(self, z) -> Any:
        z = np.array(z).astype('complex128')

        return np.piecewise(z,
                           [np.logical_and(z >= z_start, z <= z_stop) for z_start, z_stop in self.channel_limits.values()],
                           self.channel_eigenfunctions)
    
    def normalization_factor(self):

        normalization_factor = scipy.integrate.quad(lambda y: np.abs(self(y))**2, self.min_position, self.max_position, epsabs=0)[0]
        sign = np.sign(self.coefficients[0][0].real)
        return np.sqrt(normalization_factor)*sign

    @staticmethod
    def basis(k, z):
        return np.array([np.exp(-2j * np.pi * k * z), np.exp(2j * np.pi * k * z)])
    
# class FeedlineEigenfunction(Eigenfunction):
    