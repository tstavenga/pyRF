class Resonator:
    def __init__(self, resonator_network):
        self.resonator_network = resonator_network

    # def get_eigenvalue(self):


class SignalPath:
    def __init__(self, signal_path_network):
        self.signal_path_network = signal_path_network

class EigenfunctionContinuous:
    def __init__(self, coefficients, limits):
        self.coefficients = coefficients
        self.limits = limits

    def plot(self):
        zvals = np.linspace(self.starting_position, self.starting_position + self.length, 100, dtype=np.complex128)
        fig, ax = plt.subplots(1, 1)
        self.get_eigenvalue()
        resonator_eigenfunction = self.get_eigenfunction()
        ax.plot(zvals, np.abs(resonator_eigenfunction(zvals)))

