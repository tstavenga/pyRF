from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import scipy


class Eigenfunction:
    def __init__(self, 
                 coefficients: np.ndarray, 
                 channel_limits: dict, 
                 resonance_frequency: float,
                 phase_velocity: float,
                 min_position: float,
                 max_position: float) -> None:
        
        self.coefficients = coefficients
        self.channel_limits = channel_limits
        self.resonance_frequency = resonance_frequency
        self.phase_velocity = phase_velocity
        self.min_position = min_position
        self.max_position = max_position

        self.channel_eigenfunctions = self.get_channel_eigenfunctions()
        normalization_factor = self.normalization_factor()

        # normalize the eigenfunction coefficients
        self.coefficients /= normalization_factor

        # update the eigenfunctions with the new normalization factor
        self.channel_eigenfunctions = self.get_channel_eigenfunctions()
            
    def get_channel_eigenfunctions(self):
        channel_eigenfunctions = []
        
        for channel in self.channel_limits.keys():        
            channel_eigenfunctions.append(lambda z,channel=channel: np.dot(self.coefficients[2 * channel: 2 * (channel + 1)], 
                                                                                self.basis(self.resonance_frequency, self.phase_velocity, z)))
        return channel_eigenfunctions

    def __call__(self, z) -> Any:
        z = np.array(z).astype('complex128')

        return np.piecewise(z,
                           [np.logical_and(z >= z_start, z <= z_stop) for z_start, z_stop in self.channel_limits.values()],
                           self.channel_eigenfunctions)
    
    def __add__(self, other):
        # assuming that all arguments of both eigenfunctions are the same
        if self.__class__ != other.__class__:
            raise ValueError(f'Operator undefined for {self.__name__} and {other.__name__}')
        sum_eigenfunction = self.__class__(self.coefficients + other.coefficients,
                                          self.channel_limits,
                                          self.resonance_frequency_k,
                                          self.max_position,
                                          self.min_position)
        return sum_eigenfunction
    

    
    def normalization_factor(self):

        normalization_factor = scipy.integrate.quad(lambda y: np.abs(self(y))**2, self.min_position, self.max_position, epsabs=0)[0]
        sign = np.sign(self.coefficients[0].real)
        return np.sqrt(normalization_factor)*sign
    
    def plot(self, ax=None, number_of_points = 200, **plot_kwargs):
        if not ax:
            fig, ax = plt.subplots(1,1)
        position_values = np.linspace(self.min_position, self.max_position, number_of_points, dtype=np.complex128)
        eigenfunction_values = self(position_values)
        ax.plot(position_values, np.real(eigenfunction_values), **plot_kwargs, label='real part')
        ax.plot(position_values, np.imag(eigenfunction_values), **plot_kwargs, label='imaginary part')
        ax.legend()
        return ax


    @staticmethod
    def basis(frequency, phase_velocity, z):
        return np.array([np.exp(-2j * np.pi * frequency / phase_velocity * z), np.exp(2j * np.pi * frequency / phase_velocity * z)])
    
class FeedlineEigenfunction(Eigenfunction):
    def normalization_factor(self):
        return 1
    
    def orthogonalize(self, other):
        index_a = np.argmax([np.abs(self.coefficients[0]),
                          np.abs(other.coefficients[0])])
        index_b = np.argmin([np.abs(self.coefficients[0]),
                          np.abs(other.coefficients[0])])
        
        eigenfunction_a = [self, other][index_a]
        eigenfunction_b = [self, other][index_b]

        eigenfunction_a.coefficients /= eigenfunction_a.coefficients[0]
        eigenfunction_b.coefficients -= eigenfunction_a.coefficients*eigenfunction_b.coefficients[0]

        eigenfunction_b.coefficients /= eigenfunction_b.coefficients[-1]
        eigenfunction_a.coefficients -= eigenfunction_b.coefficients*eigenfunction_a.coefficients[-1]




    