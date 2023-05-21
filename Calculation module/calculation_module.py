from scipy.optimize import minimize
from scipy.linalg import null_space
import networkx as nx
import numpy as np
import re
import matplotlib.pyplot as plt
import scattering_matrix as sm


# class chip:
#     # Resonator has an eigenmode chip doesnt
#     G = nx.DiGraph()

class Resonator:
    # general mathematical functions
    # values are know constants set by the user
    # params are used for optimization to calculate the right value

    def __init__(self):
        self.R = nx.DiGraph()
        self.N_channels = 0
        self.compiled = False

    @staticmethod
    def basis(k, z):
        return np.array([np.exp(1j * k * z), np.exp(-1j * k * z)])

    @staticmethod
    def get_channel_S_matrix(dir_in, dir_out):
        S_channel = np.zeros((2, 2))
        dir_out = 1 - dir_out
        S_channel[dir_out, dir_in] = 1
        return S_channel

    def add_short(self, Anchor, z):
        short = sm.ShortMatrix(z)
        self.R.add_node(Anchor, S_matrix=short)
        return self

    def add_inductive_short(self):
        pass

    def add_open(self, Anchor, z):
        open = sm.OpenMatrix(z)
        self.R.add_node(Anchor, S_matrix=open)
        return self

    def add_capacitive_open(self, Anchor, C):
        capacitive_open_dict = {'params': 'k',
                                'values': {'C': C},
                                'S_matrix': self.capacitance_S_matrix}
        self.R.add_node(Anchor, **capacitive_open_dict)
        return self

    def add_inductive_short(self, Anchor, L):
        inductive_short_dict = {'params': 'k',
                                'values': {'L': L},
                                'S_matrix': self.inductance_S_matrix}
        self.R.add_node(Anchor, **inductive_short_dict)
        return self

    # Multiterminal scattering matrices
    def add_junction(self, Anchor, z):
        two_terminal_junction_dict = {'params': 'k',
                                      'values': {'z': z},
                                      'S_matrix': np.array([[0, 1], [1, 0]]),
                                      'channels': np.ma.masked_all(2, dtype=np.int),
                                      'directions': np.ma.masked_all(2, dtype=np.int)}
        self.R.add_node(Anchor, **two_terminal_junction_dict)
        return self

    def add_segment(self, Anchors: tuple,
                    Z0: float = None,
                    Phi0: float = None,
                    lc: float = None,
                    cc: float = None):
        self.Phi0 = Phi0
        segment_dict = dict(characteristic_values={
            'Z0': Z0,
            'Phi0': Phi0})
        print(self.R.node())
        channel_nr = self.N_channels
        nodePattern = re.compile(r"(^[A-Z]*[A-Za-z0-9])\({1}(\d+)\){1}")
        node0 = nodePattern.search(Anchors[0])
        node1 = nodePattern.search(Anchors[1])
        if node0:
            node0Name = node0.group(1)
            node0Port = int(node0.group(2))
            self.R.node[node0Name]['channels'][node0Port] = channel_nr
            self.R.node[node0Name]['directions'][node0Port] = 1  # 0: in, 1: out
        else:
            node0Name = Anchors[0]
            self.R.node[node0Name]['channels'] = np.ma.array([channel_nr], dtype=np.int)
            self.R.node[node0Name]['directions'] = np.ma.array([1], dtype=np.int)

        if node1:
            node1Name = node1.group(1)
            node1Port = int(node1.group(2))
            self.R.node[node1Name]['channels'][node1Port] = channel_nr
            self.R.node[node1Name]['directions'][node1Port] = 0
        else:
            node1Name = Anchors[1]
            self.R.node[node1Name]['channels'] = np.ma.array([channel_nr], dtype=np.int)
            self.R.node[node1Name]['directions'] = np.ma.array([0], dtype=np.int)
        position_node1 = self.R.node[node1Name]['values']['z']
        position_node0 = self.R.node[node0Name]['values']['z']
        segment_dict['length'] = position_node1 - position_node0
        self.R.add_edge(node0Name, node1Name, **segment_dict, channels=np.ma.array([channel_nr], dtype=np.int))
        self.N_channels += 1
        return self

    def _compile_node_scattering_matrix(self, n):
        # Loop over nodes to construct the scattering matrix for each node
        # Check if the number of channels is equal to the number of edges
        # for each node check if the corresponding edge is ingoing or outgoing
        # Construct the scattering matrix for the particular node
        node_data = self.R.node[n]
        node_channels = node_data['channels']
        node_directions = node_data['directions']

        edges_both = []
        edges_both.extend(list(self.R.in_edges(n)))
        edges_both.extend(list(self.R.out_edges(n)))

        if not len(node_channels.compressed()) == len(edges_both):  # edges only counts out edges :(
            raise AttributeError('node: %s does not have all ports connected' % (n))

        def node_S_matrix(k):
            S_matrix_node = np.zeros((len(node_channels), len(node_channels), 2, 2))
            for e in edges_both:
                e_data = self.R.get_edge_data(*e)
                S_node = node_data['S_matrix']
                edge_channel = e_data['channels']
                channel_index = np.argwhere(node_channels == edge_channel).flatten()[0]
                pos = node_directions[channel_index]
                if len(node_channels) == 1:
                    pos_matrix = np.diagflat(1, (2 * pos - 1))
                    S_node_mat = S_node(k, **node_data['values'],
                                        **e_data['characteristic_values'])
                    if pos == 0:
                        S_matrix_node = np.tensordot(S_node_mat,
                                                     pos_matrix, 0)
                    else:
                        S_matrix_node = np.tensordot(np.conj(S_node_mat),
                                                     pos_matrix, 0)

                elif len(node_channels) > 1:
                    # raise NotImplementedError('FIXME: add Scattering matrix for the multijunction case that has a '
                    #                           'free parameter')
                    if callable(S_node):
                        # Create function
                        # loop over nodes this seems annoying and I dont really want to do it
                        pass
                    else:
                        for ii in range(len(node_directions)):
                            S_direction = self.get_channel_S_matrix(pos, node_directions[ii])
                            S_matrix_node[ii, channel_index] += np.tensordot(S_node[ii, channel_index], S_direction, 0)
            return S_matrix_node

        node_data['node_S_matrix'] = node_S_matrix

    def _get_scattering_matrix(self):
        # loop over nodes
        # extract the scattering matrix
        # extract the channel number
        # tensor product the two
        # Add to the scattering matrix
        for n in self.R.nodes():
            self._compile_node_scattering_matrix(n)

        def S_B_matrix(k):
            S_B = np.zeros((self.N_channels, self.N_channels, 2, 2), dtype=np.complex128)
            for n in self.R.nodes():
                node_data = self.R.node[n]
                ind = np.array(node_data['channels'])
                ind_mat = np.ix_(ind, ind)
                node_S_matrix = node_data['node_S_matrix']
                if callable(node_S_matrix):
                    S_B[ind_mat] += node_S_matrix(k)
                else:
                    S_B[ind_mat] += node_S_matrix
                # print(block_matrix(S_B[np.ix_(ind, ind)]))
            return S_B

        return S_B_matrix

    # def _get_propagation_matrix(self):
    #     # loop over edges
    #     # extract channel number
    #     # construct propagation matrix
    #     # tensor product the two
    #     # add to the propagation matrix
    #
    #     def P_matrix(k):
    #         S_P = np.zeros((self.N_channels, self.N_channels, 2, 2), dtype=np.complex128)
    #         for e in self.R.edges():
    #             e_data = self.R.get_edge_data(*e)
    #             e_channel = e_data['channels']
    #             ind_mat = np.ix_(e_channel, e_channel)
    #             length = e_data['length']
    #             S_P[ind_mat] = self.propagator(k, length)
    #         return S_P
    #
    #     return P_matrix

    def mode_condition(self, k):
        if type(k) == np.ndarray:
            k = k[0]

        S_b = block_matrix(self.S_B(k))
        mode_condition = np.linalg.det(np.subtract(np.eye(2 * self.N_channels), S_b))
        return mode_condition

    def get_eigenmode(self, guess=None):
        if guess == None:
            guess = self.get_mode_guess('k')
        self.S_B = self._get_scattering_matrix()

        result = minimize(self.mode_condition, guess)
        self.k_res = abs(result['x'][0])
        self.omega_res = self.k_res * self.Phi0
        self.f_res = self.omega_res / 2 / np.pi
        #         self.tolerance = result
        return self.k_res, self.f_res

    def get_eigenfunction(self):

        T = block_matrix(self.S_B(self.k_res))
        self.eigenfunction_coefficients = null_space(np.subtract(np.eye(2 * self.N_channels), T), rcond=1e-3)[:, 0]

        # calculate nullspace
        def eigenfunction(channel, z):
            coefficients = np.zeros(2)
            if len(self.eigenfunction_coefficients) > 2:
                coefficients = self.eigenfunction_coefficients[2 * channel:2 * channel + 2]
            else:
                coefficients = self.eigenfunction_coefficients
            print(self.eigenfunction_coefficients[2:4])
            print(coefficients)
            return list(map(lambda zv: np.sum(coefficients * self.basis(self.k_res, zv)), z))

        self.eigenfunction = eigenfunction
        return eigenfunction

    # helper functions
    def get_mode_guess(self, params):
        if params == 'k':
            return 2 * np.pi / 4 / self.length
        else:
            raise NotImplementedError('Parameter %s is not implemented yet' % (self.params))

    def plot_eigenfunction(self, n_points=100):
        node0_name, node1_name = list(self.R.edges)[0]
        position_node1 = self.R.node[node1_name]['values']['z']
        position_node0 = self.R.node[node0_name]['values']['z']
        x_vals = np.linspace(position_node0, position_node1, n_points)
        y_vals = np.absolute(self.eigenfunction(0, x_vals))
        fig, ax = plt.subplots(1, 1, dpi=100)
        ax.plot(x_vals, y_vals)
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Amplitude')


def block_matrix(A):
    if not len(A.shape) == 4:
        raise ValueError('A needs to be in block matrix form. Not: %s' % (str(A.shape)))

    B = np.swapaxes(A, 1, 2)
    dims = B.shape
    return np.reshape(B, (dims[0] * dims[1], dims[2] * dims[3]))
