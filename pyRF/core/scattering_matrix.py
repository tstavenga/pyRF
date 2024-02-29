import numpy as np

class OpenMatrix:
    @staticmethod
    def scattering_matrix(frequency, position, direction_array, phase_velocity, **_):
        return np.array([[np.exp(-4j * np.pi * direction_array[0] * frequency / phase_velocity * position)]])

    @staticmethod
    def guess_phase(**_):
        return 0


class ShortMatrix:
    @staticmethod
    def scattering_matrix(frequency, position, direction_array, phase_velocity, **_):
        return np.array([[-np.exp(-4j * np.pi * direction_array[0] * frequency / phase_velocity * position)]])

    @staticmethod
    def guess_phase(**_):
        return np.pi
    
    @staticmethod
    def derivative(frequency, position, direction_array, phase_velocity, **_):
        exponential_constant = 4j * np.pi * direction_array[0] * position
        return np.array([[np.exp(-exponential_constant * frequency / phase_velocity) * exponential_constant]])
    

class CapacitorMatrix:
    @staticmethod
    def scattering_matrix(frequency, position, direction_array, characteristic_impedance, phase_velocity, capacitance):
        zc_inv = 1j * frequency * capacitance
        if len(direction_array) == 1:
            scattering_matrix = np.array([[(1 - characteristic_impedance * zc_inv) / (1 + characteristic_impedance * zc_inv) * 
                                           np.exp(-4j * np.pi * direction_array[0] * frequency / phase_velocity * position)]])
        else:
            denominator = 1 + zc_inv*characteristic_impedance
            scattering_matrix = np.array([[-zc_inv*characteristic_impedance/denominator * np.exp(-4j * np.pi * direction_array[0] * frequency / phase_velocity * position), 
                                           1/denominator],
                                          [1/denominator, 
                                           -zc_inv*characteristic_impedance/denominator * np.exp(-4j * np.pi * direction_array[1] * frequency / phase_velocity * position)]])
        return scattering_matrix

    @staticmethod
    def guess_phase(frequency, characteristic_impedance, phase_velocity, capacitance, **_):
        return 2 * np.arctan(2 * np.pi * frequency * capacitance * characteristic_impedance)

    @staticmethod
    def derivative(frequency, position, direction_array, characteristic_impedance, phase_velocity, capacitance):
        exponential_constant = -4j * np.pi * direction_array[0] * frequency / phase_velocity * position
        zc_inv = 1j * frequency * capacitance
        zc_inv_prime = 1j * phase_velocity * capacitance
        prefactor = np.exp(exponential_constant * frequency / phase_velocity)/(1 + characteristic_impedance * zc_inv)**2
        derivative = prefactor * ((characteristic_impedance * zc_inv)**2 * exponential_constant - characteristic_impedance* zc_inv_prime - exponential_constant)
        return derivative


class PortMatrix:
    @staticmethod
    def scattering_matrix(frequency, **_):
        return np.array([[0]])
    
    @staticmethod
    def guess_phase(frequency, **_):
        return 0
    
if __name__ == '__main__':
    x = OpenMatrix()
    y = CapacitorMatrix()
    k = 100
    position = 2e-3
    direction_array = 1
    characteristic_impedance = 50
    phase_velocity = 1e8
    capacitance = 30e-15
    values = {
        'frequency': 100,
        'position': 2e-3,
        'direction_array': np.array([1]),
        'characteristic_impedance': 50,
        'phase_velocity': 1e8,
        'capacitance': 30e-15,}
    print(y.derivative(**values))