from pyRF.resonator import Resonator
from . import eigenfunction as eig
import pandas as pd
import numpy as np
import scipy

class FeedLine(Resonator):
    def __init__(self, name, number_of_channels) -> None:
        super().__init__(name, number_of_channels)

    
    def get_eigenfunction(self, k):
        eigenfunction = self.eigenfunctions.get(k, None)
        if eigenfunction:
            return eigenfunction
        
        eigenfunction_coefficients = scipy.linalg.null_space(self.matrix_condition(k), rcond=1e-7)

        eigenfunctions = []

        for vector in eigenfunction_coefficients.T:
            eigenfunctions.append(eig.FeedlineEigenfunction(vector,
                                                            self.channel_limits, 
                                                            k,
                                                            self.min_position,
                                                            self.max_position))
        if len(eigenfunction_coefficients.T)==2:
            eigenfunctions[0].orthogonalize(eigenfunctions[1])

        return eigenfunctions
    
class TimeDomainSolution:
    def __init__(self, feedline, position_values):#, initial_right_values, initial_left_values):
        self.feedline = feedline
        self.position_values = position_values
        # self.initial_right_values = initial_right_values
        # self.initial_left_values = initial_left_values
        self.k_values = np.fft.fftfreq(len(position_values),position_values[1]-position_values[0])
        eigenfunction_array = {}
        for k in self.k_values:
            eigenfunction_array[k] = self.feedline.get_eigenfunction(k)
            
        self.eigenfunction_database = pd.DataFrame(eigenfunction_array)

    def channel_coefficients_vector(self, eigenfunction_array):
        channel_coefficients_left = {}
        channel_coefficients_right = {}
        for channel in range(self.feedline.number_of_channels):
            channel_coefficients_left[channel] = [phi.coefficients[2*channel] for phi in eigenfunction_array]
            channel_coefficients_right[channel] = [phi.coefficients[2*channel+1] for phi in eigenfunction_array]


    def get_timedomain_coefficients(self, ):
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
    # def get_timedomain_amplitudes(self, position_values, initial_right_values, initial_left_values):

