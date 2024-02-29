import networkx as nx
import numpy as np
import scipy
from pyRF.core import eigenfunction as eig
from pyRF.helper.adjugate import adjugate

class Resonator:
    def __init__(self, name, number_of_channels) -> None:
        self.name = name
        self.circuit_element_dict: dict = {}
        self.number_of_channels = number_of_channels
        self.phase_velocity: dict = {}
        self.length: float = None
        self.min_position: float = None
        self.max_position: float = None
        self.channel_limits: dict = {}
        self.eigenvalues: dict = {}
        self.eigenfunctions: dict = {}

    def initialize_length(self):
        min_position = np.inf
        max_position = -np.inf
        for element in self.circuit_element_dict.values():
            position = element['element'].values_dict[element['side']]['position']
            min_position = min(min_position, position)
            max_position = max(max_position, position)

        self.min_position = min_position
        self.max_position = max_position
        self.length = max_position - min_position

    def add_circuit_element(self, single_element_dict):
        if not next(iter(single_element_dict)) in self.circuit_element_dict.keys():
            self.circuit_element_dict.update(single_element_dict)

    def set_channel_limit(self, channel_number, start_position, end_position):
        channel_limit = {
            channel_number: [
                start_position,
                end_position
            ]
        }
        self.channel_limits.update(channel_limit)

    def set_phase_velocity(self, phase_velocity, channel_number):
        self.phase_velocity[channel_number] = phase_velocity

    def scattering_matrix(self, frequency):
        scattering_matrix = np.zeros((2 * self.number_of_channels, 2 * self.number_of_channels), np.complex128)
        for element in self.circuit_element_dict.values():
            element['element'].populate_scattering_matrix(frequency, element['side'], scattering_matrix)

        return scattering_matrix
    
    def scattering_matrix_derivative(self, frequency):
        scattering_matrix_derivative = np.zeros((2 * self.number_of_channels, 2 * self.number_of_channels), np.complex128)
        for element in self.circuit_element_dict.values():
            element['element'].populate_scattering_matrix_derivative(frequency, element['side'], scattering_matrix_derivative)

        return scattering_matrix_derivative
    
    def eigenvalue_guess(self, n, frequency):
        phase = 0
        for element in self.circuit_element_dict.values():
            phase += element['element'].guess_phase(frequency, element['side'])
        return (2*np.pi*n - phase)/(4*np.pi*self.length)*element['element'].values_dict[element['side']]['phase_velocity']

    def matrix_condition(self, frequency):
        return np.subtract(self.scattering_matrix(frequency), np.eye(2 * self.number_of_channels))
    
    def mode_condition(self, frequency):
        if type(frequency) == np.ndarray:
            if len(frequency)==1:
                frequency = frequency[0]
            elif len(frequency)==2:
                frequency = frequency[0] + 1j*frequency[1]
        mode_cond = np.linalg.det(self.matrix_condition(frequency))
        return [mode_cond.real, mode_cond.imag]
    
    def matrix_condition_derivative(self, frequency):
        adjugate_matrix = adjugate(self.matrix_condition(frequency))
        derivative_matrix = self.scattering_matrix_derivative(frequency)
        matrix_condition_derivative_value = np.trace(adjugate_matrix @ derivative_matrix)
        return matrix_condition_derivative_value
    
    def jacobian(self, frequency):
        if type(frequency) == np.ndarray:
            if len(frequency)==1:
                frequency = frequency[0]
            elif len(frequency)==2:
                frequency = frequency[0] + 1j*frequency[1]
        derivative = self.matrix_condition_derivative(frequency)
        a = np.real(derivative)
        b = np.imag(derivative)
        jacobian = np.array([[a, -b],
                             [b, a]])
        return jacobian

    
    def get_eigenvalue(self, n = 1):
        guess = 0
        for i in range(10):
            guess = self.eigenvalue_guess(n,guess)

        result = scipy.optimize.root(self.mode_condition, [guess,0.])#, jac=self.jacobian)
        resonance_frequency = abs(result['x'][0])
        self.eigenvalues[n] = resonance_frequency
        return resonance_frequency
    

    def get_eigenfunction(self, n=1):
        eigenfunction = self.eigenfunctions.get(n, None)
        if eigenfunction:
            return eigenfunction
        
        resonance_frequency = self.eigenvalues.get(n, self.get_eigenvalue(n))

        eigenfunction_coefficients = scipy.linalg.null_space(self.matrix_condition(resonance_frequency), rcond=1e-7)[:, 0]

        self.eigenfunctions[n] = eig.Eigenfunction(eigenfunction_coefficients, 
                                                   self.channel_limits, 
                                                   resonance_frequency, 
                                                   self.phase_velocity,
                                                   self.min_position,
                                                   self.max_position)

        return self.eigenfunctions[n]
