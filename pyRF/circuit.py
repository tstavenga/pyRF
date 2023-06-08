from pyRF.scattering_matrix import TransmissionLine
import pyRF.node_element as node_element
from pyRF.resonator import Resonator

# from scipy.optimize import minimize

# from scipy.linalg import null_space
# import scipy.integrate
# import scipy
# import networkx as nx
# import numpy as np
# import re
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

class Circuit:
    def __init__(self, name):
        self.name = name
        self.circuit_elements: dict = None
        self.transmission_lines: dict = None
        self.circuit_element_dict: dict = dict()
        self.transmission_line_dict: dict = dict()
        self.resonator_dict: dict = dict()
        self.resonators: dict = dict()

    def define_circuit_elements(self):
        pass

    def initialize_circuit_elements(self):
        for element_name, element in self.circuit_elements.items():
            circuit_element = getattr(node_element, element['element'])
            self.circuit_element_dict[element_name] = circuit_element(element_name, **element['options'])
        return

    def define_transmission_lines(self):
        pass

    def initialize_transmission_lines(self):
        for channel_number, (transmission_line_name, transmission_line) in enumerate(self.transmission_lines.items()):
            self.transmission_line_dict[transmission_line_name] = TransmissionLine(**transmission_line, channel_number = channel_number)
        return
    
    def define_resonators(self):
        pass

    def initialize_resonators(self):
        for resonator_name, connections in self.resonators.items():
            # transmission_line = self.transmission_line_dict[resonator['transmission_line']]
            self.resonator_dict[resonator_name] = self.initialize_single_resonator(resonator_name, connections)
        return
    
    def initialize_single_resonator(self, resonator_name, connections):
        resonator = Resonator(resonator_name)
        for connection_name, connection_settings in connections.items():
            # add the node element to the resonator
            # add the transmission line settings to the node element
            
            start_element_name = connection_settings['start_pin']['element']
            start_node_element = self.circuit_element_dict[start_element_name]
            start_element = {
                start_element_name: start_node_element
            }

            end_element_name = connection_settings['end_pin']['element'] 
            end_node_element = self.circuit_element_dict[end_element_name]

            end_element = {
                end_element_name: end_node_element
            }

            resonator.add_circuit_element(start_element)
            resonator.add_circuit_element(end_element)

            # add the transmission line parameters to the correct pin of the node element
            start_pin = connection_settings['start_pin']['pin']
            end_pin = connection_settings['end_pin']['pin']

            transmission_line = connection_settings['transmission_line']

            start_node_element.set_transmission_line(start_pin, transmission_line)
            end_node_element.set_transmission_line(end_pin, transmission_line)



        return resonator
    
    def initialize(self):
        
        self.define_circuit_elements()
        self.define_transmission_lines()
        self.define_resonators()

        self.initialize_circuit_elements()
        self.initialize_transmission_lines()
        self.initialize_resonators()

        


    