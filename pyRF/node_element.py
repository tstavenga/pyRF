import numpy as np

from . import scattering_matrix as sm


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
            self.values_dict[side] = self.values
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
            


    def populate_scattering_matrix_derivative(self, k, side, scattering_matrix_derivative_total):
        channel_array = np.array(list(self.direction_dict[side].keys()))
        direction_array = np.array(list(self.direction_dict[side].values()))
        index_array_1 = direction_array + 2 * channel_array
        index_array_2 = 1 - direction_array + 2 * channel_array
        index_x, index_y = np.meshgrid(index_array_1, index_array_2)
        scattering_matrix_derivative_total[index_y, index_x] = self.scattering_matrix_dict[side].derivative(
            k, **self.values_dict[side])

    def guess_phase(self, k, side):
        return self.scattering_matrix_dict[side].guess_phase(k = k, **self.values_dict[side])

# class NodeElement:
#     def __init__(self, name):
#         self.name = name
#         self.pins = dict()
#         self.space_network = nx.DiGraph()
#         self.space_nodes = None
#         self.time_edges = None
#         self.channel_list = None
#         self.direction_list = None
#         self.scattering_matrix_dict = dict()

#     def __hash__(self):
#         return hash(self.name)

#     def get_node_name(self, node_number):
#         if self.space_nodes > 1:
#             return self.name + '_' + str(node_number)
#         elif self.space_nodes==1:
#             return self.name
#         else:
#             raise AttributeError('no space ports, this error should not happen, there is something wrong with the class')

#     def set_channel(self, port, channel):
#         if port < self.space_nodes:
#             self.channel_list[port] = channel
#         else:
#             raise ValueError('Tried setting a node with too many ports, node ports: %d, given port: %d'
#                              %(self.space_nodes, port))

#     def set_direction(self, port, direction):
#         # direction is defined as 1 for outgoing and -1 for incoming direction.
#         if port < self.space_nodes:
#             self.direction_list[port] = direction
#         else:
#             raise ValueError('Tried setting a node with too many ports, node ports: %d, given port: %d'
#                              %(self.space_nodes, port))

#     def _check_connections(self):
#         if not all([self.space_nodes, self.channel_list is not None, self.direction_list is not None]): #self.time_edges
#             return False
#         elif not (len(self.channel_list.compressed()) == self.space_nodes) and \
#                (len(self.direction_list.compressed()) == self.space_nodes):
#             return False
#         else:
#             return True


# class CouplingCapacitor(NodeElement):
#     def __init__(self, name, c):
#         super().__init__(name)
#         self.capacitance = c
#         self.space_nodes = 2
#         self.time_edges = 1
#         self.channel_list = np.ma.masked_all(self.space_nodes, dtype=np.int)
#         self.direction_list = np.ma.masked_all(self.space_nodes, dtype=np.int)
#         for sp in range(self.space_nodes):
#             node_name = name + '_' + str(sp)
#             self.scattering_matrix_dict[node_name] = sm.CapacitanceMatrix(c)
#             self.space_network.add_node(node_name)

# class GroundedCapacitor(NodeElement):
#     def __init__(self, name, position, capacitance):
#         super().__init__(name, position)
#         self.capacitance = capacitance
#         self.pins = {
#             'alpha': dict(),
#             'beta': dict(),
#             }

        # self.scattering_matrix = sm.CapacitanceMatrix(capacitance)


# class Reflector(NodeElement):
#     def __init__(self, name):
#         super().__init__(name)
#         self.space_nodes = 1
#         self.time_edges = 0
#         self.channel_list = np.ma.masked_all(self.space_nodes, dtype=np.int)
#         self.direction_list = np.ma.masked_all(self.space_nodes, dtype=np.int)
#         self.scattering_matrix_dict[name] = sm.ReflectorMatrix()
#         self.space_network.add_node(name)

# class Open(NodeElement):
#     def __init__(self, name):
#         super().__init__(name)
#         self.space_nodes = 1
#         self.time_edges = 0
#         self.channel_list = np.ma.masked_all(self.space_nodes, dtype=np.int)
#         self.direction_list = np.ma.masked_all(self.space_nodes, dtype=np.int)
#         self.scattering_matrix_dict[name] = sm.OpenMatrix()
#         self.space_network.add_node(name)

# class Short(NodeElement):
#     def __init__(self, name, position):
#         super().__init__(name, position)
#         self.pins = {'alpha': {'position': position}}
#         self.space_nodes = 1
#         self.time_edges = 0
#         self.channel_list = np.ma.masked_all(self.space_nodes, dtype=np.int)
#         self.direction_list = np.ma.masked_all(self.space_nodes, dtype=np.int)
#         self.scattering_matrix_dict[name] = sm.ShortMatrix()
#         self.space_network.add_node(name)

# class Load(NodeElement):
#     def __init__(self, name):
#         super().__init__(name)
#         self.space_nodes = 1
#         self.time_edges = 0
#         self.channel_list = np.ma.masked_all(self.space_nodes, dtype=np.int)
#         self.direction_list = np.ma.masked_all(self.space_nodes, dtype=np.int)
#         self.scattering_matrix_dict[name] = sm.LoadMatrix()
#         self.space_network.add_node(name)
