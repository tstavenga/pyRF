from pyRF.resonator import Resonator
import numpy as np
import scipy

class FeedLine(Resonator):
    def __init__(self, name, number_of_channels) -> None:
        super().__init__(name, number_of_channels)


    def get_eigenfunction(self, k):
        eigenfunction = self.eigenfunctions.get(k, None)
        if eigenfunction:
            return eigenfunction
        
        eigenfunction_coefficients = scipy.linalg.null_space(self.matrix_condition(k), rcond=1e-7)

        for vector in eigenfunction_coefficients.T:
            

        coefficients = {}
        for channel in range(self.number_of_channels):
            coefficients[channel] = eigenfunction_coefficients[2 * channel: 2 * (channel + 1)]
        self.eigenfunctions[n] = eig.Eigenfunction(coefficients, 
                                                   self.channel_limits, 
                                                   resonance_frequency_k, 
                                                   self.min_position,
                                                   self.max_position)

        return self.eigenfunctions[n]
        