from scipy.optimize import minimize
from scipy.linalg import null_space
import networkx as nx
import numpy as np
import re
import matplotlib.pyplot as plt




class ScatteringMatrix:
    def __init__(self):
        self.values = dict()
        self.scattering_matrix = None
        self.space_port_dict = dict()
        self.channel_dict = dict()
        self.transmission_line_dict = dict()
        self.direction_dict = dict()
        self.dir_dict = dict()
        self.port_dict = dict()
        self.ports = 0

    def set_position(self, z):
        self.values['z'] = z

    def get_position(self):
        return self.values['z']

    def update_values(self):
        self.values['d'] = self.dir_dict

    def set_transmision_line(self, transmission_line, direction):
        port = self.ports
        self.transmission_line_dict[port] = transmission_line
        self.direction_dict[port] = direction
        self.dir_dict[port] = 1 - 2 * direction
        self.channel_dict[port] = transmission_line.channel_nr
        self.port_dict[transmission_line.channel_nr] = port
        self.ports += 1

    def node_scattering_matrix(self, k):
        return self.scattering_matrix(k, **self.values)

    def get_indices(self):
        direction_array = np.array(list(self.direction_dict.values()))
        channel_array = np.array(list(self.channel_dict.values()))
        index_array_1 = direction_array + 2 * channel_array
        index_array_2 = 1 - direction_array + 2 * channel_array
        index_x, index_y = np.meshgrid(index_array_1, index_array_2)
        return index_x, index_y

    def get_scattering_matrix(self):
        direction_array = np.array(list(self.direction_dict.values()))
        # ind_direction = np.nonzero(direction_array)
        channel_array = np.array(list(self.channel_dict.values()))
        index_array_1 = direction_array + 2 * channel_array
        index_array_2 = 1 - direction_array + 2 * channel_array
        # index_1_conjugate = np.arange(len(channel_array))
        # index_2_conjugate = np.nonzero(direction_array)
        index_x, index_y = np.meshgrid(index_array_1, index_array_2)
        # index_x_conjugate, index_y_conjugate = np.meshgrid(index_1_conjugate, index_2_conjugate)

        def node_scattering_matrix(k):
            scattering_matrix = self.scattering_matrix(k, **self.values)

            # scattering_matrix[index_x_conjugate,index_y_conjugate] = np.conjugate(
            #     scattering_matrix[index_x_conjugate,index_y_conjugate])

            return scattering_matrix
        return index_x, index_y, node_scattering_matrix


class OpenMatrix(ScatteringMatrix):
    @staticmethod
    def open_s_matrix(k, z, d):
        return np.array([[np.exp(-4j * np.pi * d[0] * k * z)]])

    def __init__(self):
        super().__init__()
        self.scattering_matrix = self.open_s_matrix

    def guess_phase(self):
        return 0


class ShortMatrix(ScatteringMatrix):
    @staticmethod
    def short_s_matrix(k, z, d):
        return np.array([[-np.exp(-4j * np.pi * d[0] * k * z)]])

    def __init__(self):
        super().__init__()
        self.scattering_matrix = self.short_s_matrix

    def guess_phase(self):
        return np.pi



class CapacitanceMatrix(ScatteringMatrix):
    @staticmethod
    def capacitance_s_matrix(k, z, d, z0, phi, c):
        zc_inv = 2j * np.pi * k * phi * c
        return np.array([[(1 - z0 * zc_inv) / (1 + z0 * zc_inv) * np.exp(-4j * np.pi * d[0] * k * z)]])

    @staticmethod
    def capacitance_s_matrix2(k, z, d, z0, phi, c):
        zc_inv = 2j * np.pi * k * phi * c
        denominator = 1 + zc_inv*z0
        s_matrix = np.array([[-zc_inv*z0/denominator * np.exp(-4j * np.pi * d[0] * k * z), 1/denominator],
                             [1/denominator, -zc_inv*z0/denominator * np.exp(-4j * np.pi * d[1] * k * z)]])
        return s_matrix

    def __init__(self, c):
        super().__init__()
        self.values['c'] = c
        self.scattering_matrix = None

    def guess_phase(self):
        return 0

    def update_values(self):
        if len(self.channel_dict) == 0:
            return
        if len(self.channel_dict) == 1:
            self.scattering_matrix = self.capacitance_s_matrix
        if len(self.channel_dict) == 2:
            self.scattering_matrix = self.capacitance_s_matrix2

        super().update_values()

        self.values['z0'] = self.transmission_line_dict[0].z0
        self.values['phi'] = self.transmission_line_dict[0].phi


class InductanceMatrix(ScatteringMatrix):
    @staticmethod
    def inductance_s_matrix(k, z, z0, phi, L):
        zL = 1j * k * phi * L
        return (zL - z0) / (zL + z0) * np.exp(-2j * k * z)

    def __init__(self, z, L):
        super().__init__()
        self.values['L'] = L
        self.scattering_matrix = self.inductance_s_matrix

    def guess_phase(self):
        return np.pi

    def update_values(self):
        super().update_values()
        self.values['z0'] = self.transmission_line_dict[0].z0
        self.values['phi'] = self.transmission_line_dict[0].phi
        pass

class LoadMatrix(ScatteringMatrix):
    @staticmethod
    def load_s_matrix(k, z):
        return 0

    def __init__(self):
        super().__init__()
        self.scattering_matrix = self.load_s_matrix

    def guess_phase(self):
        return 0


    def get_scattering_matrix(self):
        direction_array = np.array(list(self.direction_dict.values()))
        channel_array = np.array(list(self.channel_dict.values()))
        index_array_1 = direction_array + 2 * channel_array
        index_x, index_y = np.meshgrid(index_array_1, index_array_1)

        def node_scattering_matrix(k):
            return 1

        return index_x, index_y, node_scattering_matrix




class ReflectorMatrix(ScatteringMatrix):
    @staticmethod
    def reflector_s_matrix(k, z, d):
        return np.array([[-1j*np.exp(-4j*np.pi * d[0] * k * z),   1],
                         [1,                                -1j*np.exp(-4j * np.pi* d[1] * k * z)]])/np.sqrt(2)

    def __init__(self):
        super().__init__()
        self.scattering_matrix = self.reflector_s_matrix

    def guess_phase(self):
        return 0

    def update_values(self):
        super().update_values()



class TransmissionLine:
    def __init__(self, z0, phi, length, channel_nr):
        self.z0 = z0
        self.phi = phi
        self.channel_nr = channel_nr
        self.length = length

