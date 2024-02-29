import pandas as pd
import numpy as np
import scipy
from pyRF.core.resonator import Resonator
from pyRF.core import eigenfunction as eig


class FeedLine(Resonator):
    def __init__(self, name, number_of_channels) -> None:
        super().__init__(name, number_of_channels)

    def get_eigenfunction(self, frequency):
        eigenfunction = self.eigenfunctions.get(frequency, None)
        if eigenfunction:
            return eigenfunction

        eigenfunction_coefficients = scipy.linalg.null_space(
            self.matrix_condition(frequency), rcond=1e-7)

        eigenfunctions = []

        for vector in eigenfunction_coefficients.T:
            eigenfunctions.append(
                eig.FeedlineEigenfunction(vector, self.channel_limits, frequency,
                                          self.phase_velocity,
                                          self.min_position,
                                          self.max_position))
            
        if len(eigenfunction_coefficients.T) == 2:
            eigenfunctions[0].orthogonalize(eigenfunctions[1])

        return eigenfunctions


class TimeDomainSolution:
    def __init__(self,
                 feedline: FeedLine,
                 position_values: np.ndarray,
                 initial_right_values: np.ndarray,
                 initial_left_values: np.ndarray,
                 wave_speed: float = 1e-3):
        self.feedline = feedline
        self.position_values = position_values
        self.initial_right_values = initial_right_values
        self.initial_left_values = initial_left_values
        self.wave_speed = wave_speed
        self.coefficients_left = {}
        self.coefficients_right = {}
        self.k_values = np.fft.fftfreq(len(position_values),
                                       position_values[1] - position_values[0])
        eigenfunction_array = {}
        for k in self.k_values:
            eigenfunction_array[k] = self.feedline.get_eigenfunction(k)

        self.eigenfunction_database = pd.DataFrame(eigenfunction_array)

    def channel_coefficients_vector(self, eigenfunction_array):
        channel_coefficients_left = {}
        channel_coefficients_right = {}
        for channel in range(self.feedline.number_of_channels):
            channel_coefficients_right[channel] = [
                phi.coefficients[2 * channel] for phi in eigenfunction_array
            ]
            channel_coefficients_left[channel] = [
                phi.coefficients[2 * channel + 1]
                for phi in eigenfunction_array
            ]
        return channel_coefficients_left, channel_coefficients_right

    def get_timedomain_coefficients(self, initial_values,
                                    channel_coefficients_vector, direction):
        time_domain_coefficients = np.zeros(len(initial_values),
                                            dtype=np.complex128)

        for i in range(self.feedline.number_of_channels):
            z_start, z_stop = self.feedline.channel_limits[i]
            inds = np.argwhere(
                np.logical_and(self.position_values >= z_start,
                               self.position_values <= z_stop)).flatten()

            # right moving waves
            temp_y = np.zeros(len(initial_values), dtype=np.complex128)
            temp_y[inds] = initial_values[inds]

            if direction == 'right':
                time_domain_coefficients += np.fft.fft(
                    temp_y, norm='ortho') * np.conjugate(
                        channel_coefficients_vector[i])  #zero for right
            elif direction == 'left':
                time_domain_coefficients += np.fft.ifft(
                    temp_y, norm='ortho') * np.conjugate(
                        channel_coefficients_vector[i])  #one for left

        return time_domain_coefficients

    def get_solution_k_space(self):
        number_eigenfunctions = self.eigenfunction_database.values.shape[0]

        # loop over eigenfunctions
        kspace_eigenfunctions = {}
        for i in range(number_eigenfunctions):
            self.coefficients_left[i], self.coefficients_right[i] = self.channel_coefficients_vector(
                self.eigenfunction_database.values[i, :])

            kspace_eigenfunctions[i] = self.get_timedomain_coefficients(
                self.initial_left_values, self.coefficients_left[i], 'left')

            kspace_eigenfunctions[i] += self.get_timedomain_coefficients(
                self.initial_right_values, self.coefficients_right[i], 'right')
            
        return kspace_eigenfunctions

    def time_evolution(self, kspace_eigenfunctions, time):

        result = {}
        for i, function in kspace_eigenfunctions.items():
            result[i] = function*np.exp(-2j*np.pi*self.wave_speed*self.k_values*time)

        return result

    def time_solution(self, time):
        kspace_eigenfunctions = self.get_solution_k_space()
        kspace_eigenfunctions_time = self.time_evolution(kspace_eigenfunctions, time)

        solution_region = np.zeros((self.feedline.number_of_channels,len(self.position_values)), dtype=np.complex128)
        for channel in range(self.feedline.number_of_channels):
            for function_number, function in kspace_eigenfunctions_time.items():
                solution_region[channel] += np.fft.ifft(function*self.coefficients_right[function_number][channel], norm='ortho') + \
                                    np.fft.fft(function*self.coefficients_left[function_number][channel], norm='ortho')
        return solution_region

