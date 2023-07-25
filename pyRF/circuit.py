import pyRF.node_element as ne
from pyRF.resonator import Resonator
from pyRF.feedline import FeedLine

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
        self.circuit_element_dict: dict = {}
        self.transmission_line_dict: dict = {}
        self.resonator_dict: dict = {}
        self.resonators: dict = {}
        self.feedlines: dict = {}
        self.feedline_dict: dict = {}

    def define_circuit_elements(self):
        pass

    def initialize_circuit_elements(self):
        for element_name, element in self.circuit_elements.items():
            self.circuit_element_dict[element_name] = ne.NodeElement(element_type=element['element'],
                                                                     name=element_name,
                                                                     values=element['values'])
        return

    def define_resonators(self):
        pass

    def initialize_resonators(self):
        for resonator_name, connections in self.resonators.items():
            resonator = Resonator(resonator_name,
                                  number_of_channels=len(connections))
            self.resonator_dict[resonator_name] = self.initialize_single_resonator(resonator,
                                                                                   connections)
        return

    def define_feedlines(self):
        pass

    def initialize_feedlines(self):
        for feedline_name, connections in self.feedlines.items():
            feedline = FeedLine(feedline_name,
                                number_of_channels=len(connections))
            self.feedline_dict[feedline_name] = self.initialize_single_resonator(
                feedline, connections)
        return

    def initialize_resonator_lengths(self):
        for resonator in self.resonator_dict.values():
            resonator.initialize_length()

    def initialize_single_resonator(self, resonator, connections):

        
        for channel_number, connection_settings in enumerate(connections.values()):
            # add the node element to the resonator
            # add the transmission line settings to the node element

            self.add_channel_circuit_elements(resonator, connection_settings)
            self.set_resonator_channel_limits(
                resonator, connection_settings, channel_number)

            # add the transmission line parameters to the correct pin of the node element
            self.connect_resonator_transmission_lines(
                connection_settings, channel_number)
        return resonator

    def connect_resonator_transmission_lines(self, connection_settings, channel_number):
        OUT = 1
        IN = 0

        start_element_name = connection_settings['start_pin']['element']
        start_side = connection_settings['start_pin']['side']
        start_node_element = self.circuit_element_dict[start_element_name]

        end_element_name = connection_settings['end_pin']['element']
        end_side = connection_settings['end_pin']['side']
        end_node_element = self.circuit_element_dict[end_element_name]

        start_pin = connection_settings['start_pin']['pin']

        end_pin = connection_settings['end_pin']['pin']

        start_pin_settings = {
            'direction': OUT,
            'channel_number': channel_number,
            **connection_settings['transmission_line']
        }
        end_pin_settings = {
            'direction': IN,
            'channel_number': channel_number,
            **connection_settings['transmission_line']
        }

        start_node_element.connect_transmission_line(
            start_side, start_pin, start_pin_settings)
        end_node_element.connect_transmission_line(
            end_side, end_pin, end_pin_settings)

    def add_channel_circuit_elements(self, resonator, connection_settings):
        start_element_name = connection_settings['start_pin']['element']
        start_side = connection_settings['start_pin']['side']

        start_node_element = self.circuit_element_dict[start_element_name]

        start_element = {
            start_element_name: {
                'element': start_node_element,
                'side': start_side
            }
        }

        end_element_name = connection_settings['end_pin']['element']
        end_side = connection_settings['end_pin']['side']

        end_node_element = self.circuit_element_dict[end_element_name]

        end_element = {
            end_element_name: {
                'element': end_node_element,
                'side': end_side
            }
        }

        resonator.add_circuit_element(start_element)
        resonator.add_circuit_element(end_element)

    def set_resonator_channel_limits(self, resonator, connection_settings, channel_number):
        start_element_name = connection_settings['start_pin']['element']
        start_side = connection_settings['start_pin']['side']
        start_node_element = self.circuit_element_dict[start_element_name]

        end_element_name = connection_settings['end_pin']['element']
        end_side = connection_settings['end_pin']['side']
        end_node_element = self.circuit_element_dict[end_element_name]

        if isinstance(start_node_element.values['position'], (int, float, complex)):
            start_position = start_node_element.values['position']
        else:
            start_position = start_node_element.values['position'][start_side]

        if isinstance(end_node_element.values['position'], (int, float, complex)):
            end_position = end_node_element.values['position']
        else:
            end_position = end_node_element.values['position'][end_side]
        resonator.set_channel_limit(
            channel_number, start_position, end_position)

    def initialize_values(self):
        for circuit_element in self.circuit_element_dict.values():
            circuit_element.initialize_values()

    def initialize(self):

        self.define_circuit_elements()
        self.define_resonators()
        self.define_feedlines()

        self.initialize_circuit_elements()
        self.initialize_resonators()
        self.initialize_feedlines()

        self.initialize_values()
        self.initialize_resonator_lengths()
