import numpy as np

from pyRF.core import scattering_matrix as sm
from copy import copy


class NodeElement:
    def __init__(self, element_type, name, values):
        self.number_of_connections: int = 0
        self.element_type: str = element_type
        self.values: dict = values
        self.values_dict: dict = dict()
        self.name: str = name
        self.pins: dict = dict()
        self.direction_dict: dict = dict()
        self.scattering_matrix_dict: dict = dict()

    def connect_transmission_line(self, side, pin_name, pin_settings):
        self.pins[side] = dict() if side not in self.pins else self.pins[side]
        self.pins[side][pin_name] = pin_settings

        self.direction_dict[side] = dict() if side not in self.direction_dict else self.direction_dict[side]
        self.direction_dict[side][pin_settings['channel_number']] = pin_settings['direction']
            

    def initialize_values(self):
        for side, pin in self.pins.items(): #loop over sides
            # this should also have a side dependent position update
            direction_array = 1-2*np.array(list(self.direction_dict[side].values()))
            self.values_dict[side] = copy(self.values)
            self.values_dict[side]['position'] = self.values['position'] if isinstance(self.values['position'], (int, float, complex)) else self.values['position'][side]

            for pin_values in pin.values():
                # this only works if both channels have the same characteristic impedance and phase velocity
                # this is done to add the phase velocity and characteristic impedance to the dictionary for the 
                # scattering matrix parameters
                self.values_dict[side].update({'characteristic_impedance': pin_values['characteristic_impedance']})        
                self.values_dict[side].update({'phase_velocity': pin_values['phase_velocity']}) 
                

            self.values_dict[side].update({'direction_array': direction_array}) 
            self.scattering_matrix_dict[side] = getattr(sm, self.element_type + 'Matrix')

    def populate_scattering_matrix(self, k, side, scattering_matrix_total):
        channel_array = np.array(list(self.direction_dict[side].keys()))
        direction_array = np.array(list(self.direction_dict[side].values()))
        index_array_1 = direction_array + 2 * channel_array
        index_array_2 = 1 - direction_array + 2 * channel_array
        index_x, index_y = np.meshgrid(index_array_1, index_array_2)
        scattering_matrix_total[index_y, index_x] = self.scattering_matrix_dict[side].scattering_matrix(
            k, **self.values_dict[side])
        if self.element_type == 'Port':
            index_x, index_y = np.meshgrid(index_array_2, index_array_2)
            scattering_matrix_total[index_y, index_x] =  1
            


    def populate_scattering_matrix_derivative(self, frequency, side, scattering_matrix_derivative_total):
        channel_array = np.array(list(self.direction_dict[side].keys()))
        direction_array = np.array(list(self.direction_dict[side].values()))
        index_array_1 = direction_array + 2 * channel_array
        index_array_2 = 1 - direction_array + 2 * channel_array
        index_x, index_y = np.meshgrid(index_array_1, index_array_2)
        scattering_matrix_derivative_total[index_y, index_x] = self.scattering_matrix_dict[side].derivative(
            frequency, **self.values_dict[side])

    def guess_phase(self, frequency, side):
        return self.scattering_matrix_dict[side].guess_phase(frequency = frequency, **self.values_dict[side])
