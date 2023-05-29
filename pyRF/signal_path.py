from pyRF.resonator import Resonator
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


