import networkx as nx
import numpy as np
import scipy
from . import eigenfunction as eig
from .helper.adjugate import adjugate

class Resonator:
    def __init__(self, name, number_of_channels) -> None:
        self.name = name
        self.circuit_element_dict: dict = {}
        self.number_of_channels = number_of_channels
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

    def scattering_matrix(self, k):
        scattering_matrix = np.zeros((2 * self.number_of_channels, 2 * self.number_of_channels), np.complex128)
        for element in self.circuit_element_dict.values():
            element['element'].populate_scattering_matrix(k, element['side'], scattering_matrix)

        return scattering_matrix
    
    def scattering_matrix_derivative(self, k):
        scattering_matrix_derivative = np.zeros((2 * self.number_of_channels, 2 * self.number_of_channels), np.complex128)
        for element in self.circuit_element_dict.values():
            element['element'].populate_scattering_matrix_derivative(k, element['side'], scattering_matrix_derivative)

        return scattering_matrix_derivative
    
    def eigenvalue_guess(self, n, k):
        phase = 0
        for element in self.circuit_element_dict.values():
            phase += element['element'].guess_phase(k, element['side'])
        return (2*np.pi*n - phase)/(4*np.pi*self.length)

    def matrix_condition(self, k):
        return np.subtract(self.scattering_matrix(k), np.eye(2 * self.number_of_channels))
    
    def mode_condition(self, k):
        if type(k) == np.ndarray:
            if len(k)==1:
                k = k[0]
            elif len(k)==2:
                k = k[0] + 1j*k[1]
        mode_cond = np.linalg.det(self.matrix_condition(k))
        return [mode_cond.real, mode_cond.imag]
    
    def matrix_condition_derivative(self, k):
        adjugate_matrix = adjugate(self.matrix_condition(k))
        derivative_matrix = self.scattering_matrix_derivative(k)
        matrix_condition_derivative_value = np.trace(adjugate_matrix @ derivative_matrix)
        return matrix_condition_derivative_value
    
    def jacobian(self, k):
        if type(k) == np.ndarray:
            if len(k)==1:
                k = k[0]
            elif len(k)==2:
                k = k[0] + 1j*k[1]
        derivative = self.matrix_condition_derivative(k)
        a = np.real(derivative)
        b = np.imag(derivative)
        jacobian = np.array([[a, -b],
                             [b, a]])
        return jacobian

    
    def get_eigenvalue(self, n = 1):
        guess = 0
        for i in range(1):
            guess = self.eigenvalue_guess(n,guess)

        result = scipy.optimize.root(self.mode_condition, [guess,0.])#, jac=self.jacobian)
        resonance_frequency_k = abs(result['x'][0])
        self.eigenvalues[n] = resonance_frequency_k
        return resonance_frequency_k
    

    def get_eigenfunction(self, n=1):
        eigenfunction = self.eigenfunctions.get(n, None)
        if eigenfunction:
            return eigenfunction
        
        resonance_frequency_k = self.eigenvalues.get(n, self.get_eigenvalue(n))

        eigenfunction_coefficients = scipy.linalg.null_space(self.matrix_condition(resonance_frequency_k), rcond=1e-7)[:, 0]


        self.eigenfunctions[n] = eig.Eigenfunction(eigenfunction_coefficients, 
                                                   self.channel_limits, 
                                                   resonance_frequency_k, 
                                                   self.min_position,
                                                   self.max_position)

        return self.eigenfunctions[n]
        
        
# class Resonator:
#     def __init__(self, name):
#         self.name = name
#         self.space_network = nx.DiGraph()
#         self.N_channels = 0
#         self.length = 0
#         self.guess_phase = 0
#         self.normalization_factor = 1
#         self.eigenmodes = list()
#         self.channel_limits = dict()
#         self.channel_coefficients = dict()
#         self.channel_eigenfunction = list()

#     def add_circuit_elements(*circuit_elements):
#         pass

#     def add_transmission_lines(*transmission_lines):
#         for i, line in enumerate(transmission_lines):
#             line.set_channel_number(i)
            

#     def add_transmission_line(self, element_0_name, node_0_number, element_1_name, node_1_number, z0, phi0, length):
#         transmission_line = TransmissionLine(z0, phi0, length, self.N_channels)

#         global element_dict
#         global scattering_matrix_dict
#         element_0 = element_dict[element_0_name]
#         node_0_name = element_0.get_node_name(node_0_number)
#         scattering_matrix_0 = scattering_matrix_dict[node_0_name]
#         element_1 = element_dict[element_1_name]
#         node_1_name = element_1.get_node_name(node_1_number)
#         scattering_matrix_1 = scattering_matrix_dict[node_1_name]

#         # set the number of current channels, which is the current channel number to the node port
#         scattering_matrix_0.set_transmision_line(transmission_line, 1)
#         scattering_matrix_1.set_transmision_line(transmission_line, 0) # 1 is outward direction, 0 is incoming direction

#         self.space_network.add_edge(node_0_name, node_1_name,
#                                     transmission_line=transmission_line)
#         self.N_channels += 1
#         return self

#     def add_coupling_capacitor(self, node_name, c):
#         capacitor = ne.Capacitor(node_name, c)
#         global element_dict
#         element_dict[node_name] = capacitor
#         global scattering_matrix_dict
#         scattering_matrix_dict.update(capacitor.scattering_matrix_dict)
#         return self

#     def add_reflector(self, node_name):
#         reflector = ne.Reflector(node_name)
#         global element_dict
#         element_dict[node_name] = reflector
#         global scattering_matrix_dict
#         scattering_matrix_dict.update(reflector.scattering_matrix_dict)
#         return self

#     def add_open(self, node_name):
#         open = ne.Open(node_name)
#         global element_dict
#         element_dict[node_name] = open
#         global scattering_matrix_dict
#         scattering_matrix_dict.update(open.scattering_matrix_dict)
#         return self

#     def add_short(self, node_name):
#         short = ne.Short(node_name)
#         global element_dict
#         element_dict[node_name] = short
#         global scattering_matrix_dict
#         scattering_matrix_dict.update(short.scattering_matrix_dict)
#         return self

#     def _assign_positions(self, starting_position=0):
#         global scattering_matrix_dict
#         self.starting_position = starting_position
#         network_start = self.space_network.to_undirected()
#         starting_node = self._get_starting_node(network_start)

#         network = self.space_network.to_undirected()
#         next_position = starting_position
#         next_node = starting_node
#         self.guess_phase += scattering_matrix_dict[next_node].guess_phase()

#         for i in range(len(network.nodes())):
#             scattering_matrix_dict[next_node].set_position(next_position)
#             edges = network.edges(next_node)
#             if not edges:
#                 break

#             e = list(edges)[0]

#             transmission_line = network.get_edge_data(*e)['transmission_line']
#             previous_node = next_node
#             if previous_node == e[0]:
#                 next_node = e[1]
#             else:
#                 assert (previous_node == e[1])
#                 next_node = e[0]

#             length = transmission_line.length
#             channel_nr = transmission_line.channel_nr
#             port = scattering_matrix_dict[next_node].port_dict[channel_nr]
#             dir = scattering_matrix_dict[next_node].dir_dict[port]
#             self.channel_limits[channel_nr] = [min(next_position, next_position + dir*length),
#                                                max(next_position, next_position + dir*length)]
#             # print(self.channel_limits[channel_nr])
#             next_position += dir*length
#             network.remove_node(previous_node)

#         self.length = next_position
#         self.guess_phase += scattering_matrix_dict[next_node].guess_phase()

#     def _get_starting_node(self, network):
#         guess_node = list(network)[0]
#         if len(network.edges(guess_node)) == 1:
#             return guess_node

#         next_node = guess_node
#         while True:
#             edges = network.edges(next_node)
#             if not edges:
#                 return next_node

#             e = list(edges)[0]
#             previous_node = next_node
#             if previous_node == e[0]:
#                 next_node = e[1]
#             else:
#                 assert (previous_node == e[1])
#                 next_node = e[0]
#             network.remove_node(previous_node)

#     def scattering_matrix(self, k):
#         global scattering_matrix_dict
#         s_matrix = np.zeros((2 * self.N_channels, 2 * self.N_channels), np.complex128)
#         for node in self.space_network.nodes():
#             id_1, id_2, scattering_matrix_node = scattering_matrix_dict[node].get_scattering_matrix()
#             s_matrix[id_1, id_2] = scattering_matrix_node(k)
#         return s_matrix

#     def mode_condition(self, k):
#         if type(k) == np.ndarray:
#             if len(k)==1:
#                 k = k[0]
#             elif len(k)==2:
#                 k = k[0] + 1j*k[1]
#         mode_cond = np.linalg.det(np.subtract(np.eye(2 * self.N_channels), self.scattering_matrix(k)))
#         return [mode_cond.real, mode_cond.imag]

#     def eigenfunction_coefficients(self, k):
#         eigenfunction_coefficients = scipy.linalg.null_space(np.subtract(np.eye(2 * self.N_channels),
#                                                                          self.scattering_matrix(k)))
#         return eigenfunction_coefficients

#     def get_eigenvalue(self):
#         guess = (2 * np.pi - self.guess_phase) / abs(2*self.length)
#         result = scipy.optimize.root(self.mode_condition, [guess/2/np.pi,0.])
#         k_res = result['x'][0]
#         self.eigenmodes.append(k_res)
#         return k_res

#     def eigenfunction(self, z):
#         z = np.array(z).astype('complex128')
#         k_res = self.eigenmodes[0]
#         eigenfunction_coefficients = scipy.linalg.null_space(np.subtract(np.eye(2 * self.N_channels),
#                                                                          self.scattering_matrix(k_res)), rcond=1e-5)[:, 0]
#         self.channel_eigenfunction = []
#         for i in range(self.N_channels):
#             self.channel_coefficients[i] = eigenfunction_coefficients[2 * i: 2 * (i + 1)]/self.normalization_factor
#             self.channel_eigenfunction.append(lambda z,i=i: np.dot(self.channel_coefficients[i], self.basis(k_res, z)))


#         return np.piecewise(z,
#                            [np.logical_and(z >= z_start, z <= z_stop) for z_start, z_stop in self.channel_limits.values()],
#                            self.channel_eigenfunction)

#     def normalize_eigenfunction(self):
#         A = scipy.integrate.quad(lambda y: np.abs(self.eigenfunction(y))**2, 0, self.length, epsabs=0)[0]
#         self.normalization_factor = np.sqrt(A)



#     @staticmethod
#     def basis(k, z):
#         return np.array([np.exp(2j * np.pi * k * z), np.exp(-2j * np.pi * k * z)])
