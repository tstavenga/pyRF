import scattering_matrix as sm
import numpy as np
import networkx as nx
# class for a single space node

class node_element:
    def __init__(self, name):
        self.__name = name
        self.space_network = nx.DiGraph()
        self.space_nodes = None
        self.time_edges = None
        self.channel_list = None
        self.direction_list = None
        self.scattering_matrix_dict = dict()

    def __hash__(self):
        return hash(self.__name)

    def get_node_name(self, node_number):
        if self.space_nodes > 1:
            return self.__name + '_' + str(node_number)
        elif self.space_nodes==1:
            return self.__name
        else:
            raise AttributeError('no space ports, this error should not happen, there is something wrong with the class')

    def set_channel(self, port, channel):
        if port < self.space_nodes:
            self.channel_list[port] = channel
        else:
            raise ValueError('Tried setting a node with too many ports, node ports: %d, given port: %d'
                             %(self.space_nodes, port))

    def set_direction(self, port, direction):
        # direction is defined as 1 for outgoing and -1 for incoming direction.
        if port < self.space_nodes:
            self.direction_list[port] = direction
        else:
            raise ValueError('Tried setting a node with too many ports, node ports: %d, given port: %d'
                             %(self.space_nodes, port))

    def _check_connections(self):
        if not all([self.space_nodes, self.channel_list is not None, self.direction_list is not None]): #self.time_edges
            return False
        elif not (len(self.channel_list.compressed()) == self.space_nodes) and \
               (len(self.direction_list.compressed()) == self.space_nodes):
            return False
        else:
            return True

    # def get_scattering_matrix(self):
    #     if self.time_edges == 0:






class Capacitor(node_element):
    def __init__(self, name, c):
        super().__init__(name)
        self.capacitance = c
        self.space_nodes = 2
        self.time_edges = 1
        self.channel_list = np.ma.masked_all(self.space_nodes, dtype=np.int)
        self.direction_list = np.ma.masked_all(self.space_nodes, dtype=np.int)
        for sp in range(self.space_nodes):
            node_name = name + '_' + str(sp)
            self.scattering_matrix_dict[node_name] = sm.CapacitanceMatrix(c)
            self.space_network.add_node(node_name)

class Reflector(node_element):
    def __init__(self, name):
        super().__init__(name)
        self.space_nodes = 1
        self.time_edges = 0
        self.channel_list = np.ma.masked_all(self.space_nodes, dtype=np.int)
        self.direction_list = np.ma.masked_all(self.space_nodes, dtype=np.int)
        self.scattering_matrix_dict[name] = sm.ReflectorMatrix()
        self.space_network.add_node(name)

class Open(node_element):
    def __init__(self, name):
        super().__init__(name)
        self.space_nodes = 1
        self.time_edges = 0
        self.channel_list = np.ma.masked_all(self.space_nodes, dtype=np.int)
        self.direction_list = np.ma.masked_all(self.space_nodes, dtype=np.int)
        self.scattering_matrix_dict[name] = sm.OpenMatrix()
        self.space_network.add_node(name)

class Short(node_element):
    def __init__(self, name):
        super().__init__(name)
        self.space_nodes = 1
        self.time_edges = 0
        self.channel_list = np.ma.masked_all(self.space_nodes, dtype=np.int)
        self.direction_list = np.ma.masked_all(self.space_nodes, dtype=np.int)
        self.scattering_matrix_dict[name] = sm.ShortMatrix()
        self.space_network.add_node(name)

class Load(node_element):
    def __init__(self, name):
        super().__init__(name)
        self.space_nodes = 1
        self.time_edges = 0
        self.channel_list = np.ma.masked_all(self.space_nodes, dtype=np.int)
        self.direction_list = np.ma.masked_all(self.space_nodes, dtype=np.int)
        self.scattering_matrix_dict[name] = sm.LoadMatrix()
        self.space_network.add_node(name)











