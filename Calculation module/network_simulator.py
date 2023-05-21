from scipy.optimize import minimize
from scipy.linalg import null_space
import scipy.integrate
import scipy
import networkx as nx
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scattering_matrix as sm
import node_element as ne
import signal_path as sp


global element_dict
global scattering_matrix_dict

# The goal of this module is to manage the time and space domain couplings. this is the interface to user, which is
# expected to add capacitors/inductors/junctions etc

class Circuit:
    def __init__(self, name='Default'):
        self.name = name
        self.space_network = nx.DiGraph()
        self.time_network = nx.Graph()
        self.N_channels = 0
        self.compiled = False
        if 'element_dict' in globals():
            raise AttributeError('There can only be one instance of a circuit at a time')
        global element_dict
        element_dict = dict()
        global scattering_matrix_dict
        scattering_matrix_dict = dict()
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




    def _compile_scattering_matrix2(self):
        # first separate all the different space trees from the forest
        # then for every node in a space tree get the total scattering matrix
        # every space tree has a time domain amplitude and a seperate k vector

        # separate the unconnected space graphs

        self.resonator_list = [sp.Resonator(self.space_network.subgraph(c).copy())
                               for c in nx.weakly_connected_components(self.space_network)]

        # print(S)
        # H = G.copy()
        # edges_to_remove = [e for e in H.edges(data=True) if not e['attribute'] == 'attribute1']
        # H.remove_edges_from(edges_to_remove)
        G = self.resonator_list[0].resonator_network
        pos = nx.spring_layout(G)
        # nx.draw_spring(self.space_network, pos)
        nx.draw_networkx(G, pos, width=3, edge_color="r", alpha=0.1)
        ax = plt.gca()
        fig = plt.gcf()
        trans = ax.transData.transform
        trans2 = fig.transFigure.inverted().transform
        imsize = 0.1  # this is the image size
        for z, n in enumerate(G.nodes()):
            (x, y) = pos[n]
            xx, yy = trans((x, y))  # figure coordinates
            xa, ya = trans2((xx, yy))  # axes coordinates
            ax.plot(x, y, '*', markersize=25)

            # a = plt.axes([xa - imsize / 2.0, ya - imsize / 2.0, imsize, imsize])
            # a.imshow(img[n])
            # a.set_aspect('equal')
            # a.axis('off')
        plt.show()
        # k=0
        # for node_name, node in self.element_dict.items():
        #     node._get_scattering_matrix(k)


class Resonator:
    def __init__(self, name):
        self.name = name
        self.space_network = nx.DiGraph()
        self.N_channels = 0
        self.length = 0
        self.guess_phase = 0
        self.normalization_factor = 1
        self.eigenmodes = list()
        self.channel_limits = dict()
        self.channel_coefficients = dict()
        self.channel_eigenfunction = list()

    def add_transmission_line(self, element_0_name, node_0_number, element_1_name, node_1_number, z0, phi0, length):
        transmission_line = sm.TransmissionLine(z0, phi0, length, self.N_channels)

        global element_dict
        global scattering_matrix_dict
        element_0 = element_dict[element_0_name]
        node_0_name = element_0.get_node_name(node_0_number)
        scattering_matrix_0 = scattering_matrix_dict[node_0_name]
        element_1 = element_dict[element_1_name]
        node_1_name = element_1.get_node_name(node_1_number)
        scattering_matrix_1 = scattering_matrix_dict[node_1_name]

        # set the number of current channels, which is the current channel number to the node port
        scattering_matrix_0.set_transmision_line(transmission_line, 1)
        scattering_matrix_1.set_transmision_line(transmission_line, 0) # 1 is outward direction, 0 is incoming direction

        self.space_network.add_edge(node_0_name, node_1_name,
                                    transmission_line=transmission_line)
        self.N_channels += 1
        return self

    def add_coupling_capacitor(self, node_name, c):
        capacitor = ne.Capacitor(node_name, c)
        global element_dict
        element_dict[node_name] = capacitor
        global scattering_matrix_dict
        scattering_matrix_dict.update(capacitor.scattering_matrix_dict)
        return self

    def add_reflector(self, node_name):
        reflector = ne.Reflector(node_name)
        global element_dict
        element_dict[node_name] = reflector
        global scattering_matrix_dict
        scattering_matrix_dict.update(reflector.scattering_matrix_dict)
        return self

    def add_open(self, node_name):
        open = ne.Open(node_name)
        global element_dict
        element_dict[node_name] = open
        global scattering_matrix_dict
        scattering_matrix_dict.update(open.scattering_matrix_dict)
        return self

    def add_short(self, node_name):
        short = ne.Short(node_name)
        global element_dict
        element_dict[node_name] = short
        global scattering_matrix_dict
        scattering_matrix_dict.update(short.scattering_matrix_dict)
        return self

    def _assign_positions(self, starting_position=0):
        global scattering_matrix_dict
        self.starting_position = starting_position
        network_start = self.space_network.to_undirected()
        starting_node = self._get_starting_node(network_start)

        network = self.space_network.to_undirected()
        next_position = starting_position
        next_node = starting_node
        self.guess_phase += scattering_matrix_dict[next_node].guess_phase()

        for i in range(len(network.nodes())):
            scattering_matrix_dict[next_node].set_position(next_position)
            edges = network.edges(next_node)
            if not edges:
                break

            e = list(edges)[0]

            transmission_line = network.get_edge_data(*e)['transmission_line']
            previous_node = next_node
            if previous_node == e[0]:
                next_node = e[1]
            else:
                assert (previous_node == e[1])
                next_node = e[0]

            length = transmission_line.length
            channel_nr = transmission_line.channel_nr
            port = scattering_matrix_dict[next_node].port_dict[channel_nr]
            dir = scattering_matrix_dict[next_node].dir_dict[port]
            self.channel_limits[channel_nr] = [min(next_position, next_position + dir*length),
                                               max(next_position, next_position + dir*length)]
            # print(self.channel_limits[channel_nr])
            next_position += dir*length
            network.remove_node(previous_node)

        self.length = next_position
        self.guess_phase += scattering_matrix_dict[next_node].guess_phase()


    def _get_starting_node(self, network):
        guess_node = list(network)[0]
        if len(network.edges(guess_node)) == 1:
            return guess_node

        next_node = guess_node
        while True:
            edges = network.edges(next_node)
            if not edges:
                return next_node

            e = list(edges)[0]
            previous_node = next_node
            if previous_node == e[0]:
                next_node = e[1]
            else:
                assert (previous_node == e[1])
                next_node = e[0]
            network.remove_node(previous_node)

    def scattering_matrix(self, k):
        global scattering_matrix_dict
        s_matrix = np.zeros((2 * self.N_channels, 2 * self.N_channels), np.complex128)
        for node in self.space_network.nodes():
            id_1, id_2, scattering_matrix_node = scattering_matrix_dict[node].get_scattering_matrix()
            s_matrix[id_1, id_2] = scattering_matrix_node(k)
        return s_matrix

    def mode_condition(self, k):
        if type(k) == np.ndarray:
            if len(k)==1:
                k = k[0]
            elif len(k)==2:
                k = k[0] + 1j*k[1]
        mode_cond = np.linalg.det(np.subtract(np.eye(2 * self.N_channels), self.scattering_matrix(k)))
        return [mode_cond.real, mode_cond.imag]

    def eigenfunction_coefficients(self, k):
        eigenfunction_coefficients = scipy.linalg.null_space(np.subtract(np.eye(2 * self.N_channels),
                                                                         self.scattering_matrix(k)))
        return eigenfunction_coefficients

    def get_eigenvalue(self):
        guess = (2 * np.pi - self.guess_phase) / abs(2*self.length)
        result = scipy.optimize.root(self.mode_condition, [guess/2/np.pi,0.])
        print(result)
        k_res = result['x'][0]
        self.eigenmodes.append(k_res)
        return k_res

    def eigenfunction(self, z):
        z = np.array(z).astype('complex128')
        k_res = self.eigenmodes[0]
        eigenfunction_coefficients = scipy.linalg.null_space(np.subtract(np.eye(2 * self.N_channels),
                                                                         self.scattering_matrix(k_res)), rcond=1e-5)[:, 0]
        self.channel_eigenfunction = []
        print(self.normalization_factor)
        for i in range(self.N_channels):
            self.channel_coefficients[i] = eigenfunction_coefficients[2 * i: 2 * (i + 1)]/self.normalization_factor
            self.channel_eigenfunction.append(lambda z,i=i: np.dot(self.channel_coefficients[i], self.basis(k_res, z)))


        return np.piecewise(z,
                           [np.logical_and(z >= z_start, z <= z_stop) for z_start, z_stop in self.channel_limits.values()],
                           self.channel_eigenfunction)

    def normalize_eigenfunction(self):
        A = scipy.integrate.quad(lambda y: np.abs(self.eigenfunction(y))**2, 0, self.length, epsabs=0)[0]
        self.normalization_factor = np.sqrt(A)



    @staticmethod
    def basis(k, z):
        return np.array([np.exp(2j * np.pi * k * z), np.exp(-2j * np.pi * k * z)])

    def plot_eigenfunction(self):
        global scattering_matrix_dict
        zvals = np.linspace(self.starting_position,self.starting_position+self.length, 100, dtype=np.complex128)
        fig,ax = plt.subplots(1,1)
        self.get_eigenvalue()
        # resonator_eigenfunction = self.eigenfunction()
        ax.plot(zvals,np.abs(self.eigenfunction(zvals)))
        nodes = self.space_network.to_undirected().nodes()
        for node in nodes:
            # print(node + " position: ",scattering_matrix_dict[node].get_position())
            xp = scattering_matrix_dict[node].get_position()
            xp_c = np.array(xp).astype(np.complex128)
            yp = np.abs(self.eigenfunction(xp_c))
            ax.plot(xp,yp,'r.', markersize=10)
            ax.text(xp,
                    yp,
                    node,
                    horizontalalignment='left',
                    verticalalignment='top')




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











































class SignalPath(Resonator):
    def __init__(self, name):
        super().__init__(name)

    def add_matched_load(self, node_name):
        load = ne.Load(node_name)
        global element_dict
        element_dict[node_name] = load
        global scattering_matrix_dict
        scattering_matrix_dict.update(load.scattering_matrix_dict)
        return self

    def get_eigenvalue(self):
        pass

    def eigenfunction_coefficients(self, k):
        eigenfunction_coefficients = scipy.linalg.null_space(np.subtract(np.eye(2 * self.N_channels),
                                                                         self.scattering_matrix(k)))
        return eigenfunction_coefficients

    def eigenfunction_coefficients_g(self, k):
        # eigenfunction_coefficients = scipy.linalg.orth(self.scattering_matrix(k))
        eigenfunction_coefficients = scipy.linalg.null_space(np.subtract(np.eye(2 * self.N_channels),
                                                                         self.scattering_matrix(k)))
        for i in range(eigenfunction_coefficients.shape[1]):
            phi = eigenfunction_coefficients[:, i]
            yield phi

    def eigenfunction_coefficients_v(self, k):
        dims = scipy.linalg.null_space(np.subtract(np.eye(2 * self.N_channels),
                                            self.scattering_matrix(k[0]))).shape
        # dims = scipy.linalg.orth(self.scattering_matrix(k[0]),rcond=1e-5).shape
        eigenfunction_coefficients_v = np.zeros((dims[1],dims[0], len(k)), dtype=np.complex128)

        for i, kc in enumerate(k):
            eigenfunction_coefficients = scipy.linalg.null_space(np.subtract(np.eye(2 * self.N_channels),
                                                self.scattering_matrix(kc)))
            Aind = np.argmax(np.abs(eigenfunction_coefficients[0, :]))
            A = eigenfunction_coefficients[:, Aind] / eigenfunction_coefficients[0, Aind]
            # Bind = np.argmax(np.abs(eigenfunction_coefficients[-1, :]))

            B = eigenfunction_coefficients[:, 1-Aind]
            B = B - A * B[0]
            B = B / B[-1]
            A = A - B * A[-1]
            eigenfunction_coefficients_new = np.array([A,B]).T
            for j in range(eigenfunction_coefficients_new.shape[1]): # loop over eigenfunctions
                eigenfunction_coefficients_v[j,:,i] = eigenfunction_coefficients_new[:,j]
            # if i!=0: #check if the order of the eigenfunctions is consistent
            #     phi = eigenfunction_coefficients_v[0,:,i]
            #     psi = eigenfunction_coefficients_v[0,:,i-1]
            #     overlap = phi.dot(np.conjugate(psi))
            #     if np.abs(overlap)<0.5: #swap the order of the i-th eigenfunction
            #         temp = np.copy(eigenfunction_coefficients_v[0,:,i])
            #         eigenfunction_coefficients_v[0, :, i] = eigenfunction_coefficients_v[1,:,i]
            #         eigenfunction_coefficients_v[1, :, i] = temp

        return eigenfunction_coefficients_v


    def eigenfunctions(self, k, z):
        for phi_flat in self.eigenfunction_coefficients_g(k):
            phi = np.reshape(phi_flat, (self.N_channels,2))
            y = np.zeros(len(z),dtype=type(z))
            for i in range(self.N_channels):
                z_start, z_stop = self.channel_limits[i]
                inds = np.argwhere(np.logical_and(z>=z_start, z<=z_stop)).flatten()
                y[inds] = np.dot(phi[i], self.basis(k, z[inds]))
            yield y

    def solution(self, z, yleft, yright, t, c=5e-3):
        k = np.fft.fftfreq(len(z),z[1]-z[0])
        eigenfunctions = self.eigenfunction_coefficients_v(k)
        A1left = np.zeros(len(z), dtype=np.complex128)
        A1right = np.zeros(len(z), dtype=np.complex128)
        phi = np.reshape(eigenfunctions,(len(eigenfunctions),(self.N_channels),2,len(k)))
        phi1 = phi[0,:,:,:]
        for i in range(self.N_channels):
            z_start, z_stop = self.channel_limits[i]
            inds = np.argwhere(np.logical_and(z >= z_start, z <= z_stop)).flatten()

            # right moving waves
            temp_y = np.zeros(len(yright),dtype=np.complex128)
            temp_y[inds] = yright[inds]
            A1right += np.fft.fft(temp_y,norm='ortho')*np.conjugate(phi1[i,0,:]) #zero for right

            #left moving waves
            temp_y = np.zeros(len(yright),dtype=np.complex128)
            temp_y[inds] = yleft[inds]
            A1left += np.fft.ifft(temp_y, norm='ortho')*np.conjugate(phi1[i,1,:]) #one for left

        phi2 = phi[1,:,:,:]
        B1left = np.zeros(len(z), dtype=np.complex128)
        B1right = np.zeros(len(z), dtype=np.complex128)
        for i in range(self.N_channels):
            z_start, z_stop = self.channel_limits[i]
            inds = np.argwhere(np.logical_and(z >= z_start, z <= z_stop)).flatten()

            # right moving waves
            temp_y = np.zeros(len(yright),dtype=np.complex128)
            temp_y[inds] = yright[inds]
            B1right += np.fft.fft(temp_y,norm='ortho')*np.conjugate(phi2[i,0,:]) #zero for right

            #left moving waves
            temp_y = np.zeros(len(yright),dtype=np.complex128)
            temp_y[inds] = yleft[inds]
            B1left += np.fft.ifft(temp_y, norm='ortho')*np.conjugate(phi2[i,1,:]) #one for left

        # A1 = np.fft.ifft(A1left, norm='ortho') + np.fft.fft(A1right, norm='ortho')
        A1t = (A1left+A1right)*np.exp(-2j*np.pi*c*k*t)
        B1t = (B1left+B1right)*np.exp(-2j*np.pi*c*k*t)



        solution_region = np.zeros((self.N_channels,len(z)), dtype=np.complex128)
        for i in range(self.N_channels):
            solution_region[i] += np.fft.ifft(A1t*phi1[i,0,:], norm='ortho') + \
                                  np.fft.fft(A1t*phi1[i,1,:], norm='ortho') + \
                                  np.fft.ifft(B1t * phi2[i, 0, :], norm='ortho') + \
                                  np.fft.fft(B1t * phi2[i, 1, :], norm='ortho')
        return solution_region

    def solution_animation(self,z, yleft, yright, times, fig = None, ax = None, c=5e-3):
        if not fig or not ax:
            fig,ax = plt.subplots(1,1)
        sol = self.solution(z, yleft, yright, t=times[0],c=c)
        lines = []
        for s in sol:
            line, = ax.plot(z, abs(s))
            lines.append(line)

        def frame(t):
            sols = self.solution(z, yleft, yright, t=t,c=c)
            for i, (line, sol) in enumerate(zip(lines,sols)):
                z_start, z_stop = self.channel_limits[i]
                inds = np.argwhere(np.logical_and(z >= z_start, z <= z_stop)).flatten()
                line.set_data(z[inds], abs(sol[inds]))
            return lines
        anim = animation.FuncAnimation(fig, frame, frames=times, interval=50, blit=True)
        plt.show()
        return

    def get_eigenfunctions(self):
        res_matrix = self.scattering_matrix(k_res)
        # eigenfunction_coefficients = scipy.linalg.orth(res_matrix)
        eigenfunction_coefficients = null_space(np.subtract(np.eye(2 * self.N_channels), res_matrix))[:,0]
        self.eigenfunction_coefficients.append(eigenfunction_coefficients)
        for i in range(self.N_channels):

            self.channel_coefficients[i] = eigenfunction_coefficients[2 * i: 2 * (i + 1)]
            self.channel_eigenfunction.append(lambda z: np.dot(self.channel_coefficients[i], self.basis(k_res, z)))

        def eigenfunction(z):

            return np.piecewise(z,
                                [np.logical_and(z >= z_start, z <= z_stop) for z_start, z_stop in
                                 self.channel_limits.values()],
                                self.channel_eigenfunction)

        self.eigenfunctions.append(eigenfunction)

        return eigenfunction


    # def get_eigenvalue(self):

# This code snippet will be useful for plotting the final eigenfunctions
#
# t = np.linspace(0, 10, 200)
# x = np.cos(np.pi * t)
# y = np.sin(t)
#
# # Create a set of line segments so that we can color them individually
# # This creates the points as a N x 1 x 2 array so that we can stack points
# # together easily to get the segments. The segments array for line collection
# # needs to be numlines x points per line x 2 (x and y)
# points = np.array([x, y]).T.reshape(-1, 1, 2)
# segments = np.concatenate([points[:-1], points[1:]], axis=1)
#
# # Create the line collection object, setting the colormapping parameters.
# # Have to set the actual values used for colormapping separately.
# lc = LineCollection(segments, cmap=plt.get_cmap('copper'),
#     norm=plt.Normalize(0, 10))
# lc.set_array(t)
# lc.set_linewidth(3)
