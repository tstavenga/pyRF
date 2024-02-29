from scipy.optimize import minimize
from scipy.linalg import null_space
import scipy.integrate
import scipy
import networkx as nx
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation


global element_dict
global scattering_matrix_dict

# The goal of this module is to manage the time and space domain couplings. this is the interface to user, which is
# expected to add capacitors/inductors/junctions etc

class Circuit:
    def __init__(self, name='Default'):
        self.name = name
        # self.space_network = nx.DiGraph()
        # self.time_network = nx.Graph()
        self.N_channels = 0
        self.compiled = False
        # if 'element_dict' in globals():
        #     raise AttributeError('There can only be one instance of a circuit at a time')
        self.element_dict = dict()
        # self.scattering_matrix_dict
        # scattering_matrix_dict = dict()
        self.resonator_dict = {}
        self.signal_path_dict = {}

    def add_resonator(self, resonator):
        self.resonator_dict[resonator.name] = resonator
        self.space_network.add_nodes_from(resonator.space_network.nodes(data=True))
        self.space_network.add_edges_from(resonator.space_network.edges(data=True))

    def add_signal_path(self, signal_path):
        self.signal_path_dict[signal_path.name] = signal_path
        self.space_network.add_edges_from(signal_path.space_network.edges(data=True))
        self.space_network.add_nodes_from(signal_path.space_network.nodes(data=True))

    def  finish(self, starting_position=0):
        self._compile_scattering_matrix_dict()
        self._check_connections()
        self._assign_positions(starting_position)
        self._update_values()
        # self._compile_scattering_matrix()
        return self


    def _update_values(self):
        global scattering_matrix_dict
        for smd in scattering_matrix_dict.values():
            smd.update_values()

    def _compile_scattering_matrix_dict(self):
        global element_dict
        global scattering_matrix_dict
        for node_name, element in element_dict.items():
            scattering_matrix_dict.update(element.scattering_matrix_dict)

    def _check_connections(self):
        global element_dict
        for node_name, element in element_dict.items():
            element._check_connections()

    def _assign_positions(self, starting_position = 0):
        for resonator in self.resonator_dict.values():
            resonator._assign_positions(starting_position = starting_position)

        for signal_path in self.signal_path_dict.values():
            signal_path._assign_positions(starting_position = starting_position)

    def time_domain_coupling(self, signal_path_name, resonator_name, coupling_capacitor_name):
        # get the coupling capacitor
        resonator = resonator_dict[resonator_name]
        feedline = signal_path_dict[signal_path_name]

        # based on the eigenfunctions calculate the coupling at the resonator frequency




def assign_next_position(network, next_node, next_position):
    global scattering_matrix_dict
    scattering_matrix_dict[next_node].set_position(next_position)
    edges = network.edges(next_node)
    if not edges:
        return

    e = list(edges)[0]

    transmission_line = network.get_edge_data(*e)['transmission_line']
    previous_node = next_node
    if previous_node == e[0]:
        next_node = e[1]
    else:
        assert (previous_node == e[1])
        next_node = e[0]

    length = transmission_line.length
    next_position += length
    network.remove_node(previous_node)

    assign_next_position(network, next_node, next_position)


