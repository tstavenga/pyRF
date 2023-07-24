import numpy as np




class OpenMatrix:
    @staticmethod
    def scattering_matrix(k, position, direction_array, **_):
        return np.array([[np.exp(-4j * np.pi * direction_array[0] * k * position)]])

    @staticmethod
    def guess_phase(**_):
        return 0


class ShortMatrix:
    @staticmethod
    def scattering_matrix(k, position, direction_array, **_):
        return np.array([[-np.exp(-4j * np.pi * direction_array[0] * k * position)]])

    @staticmethod
    def guess_phase(**_):
        return np.pi
    

class CapacitorMatrix:
    @staticmethod
    def scattering_matrix(k, position, direction_array, characteristic_impedance, phase_velocity, capacitance):
        zc_inv = 2j * np.pi * k * phase_velocity * capacitance
        if len(direction_array) == 1:
            scattering_matrix = np.array([[(1 - characteristic_impedance * zc_inv) / (1 + characteristic_impedance * zc_inv) * 
                                           np.exp(-4j * np.pi * direction_array[0] * k * position)]])
        else:
            denominator = 1 + zc_inv*characteristic_impedance
            scattering_matrix = np.array([[-zc_inv*characteristic_impedance/denominator * np.exp(-4j * np.pi * direction_array[0] * k * position), 
                                           1/denominator],
                                          [1/denominator, 
                                           -zc_inv*characteristic_impedance/denominator * np.exp(-4j * np.pi * direction_array[1] * k * position)]])
        return scattering_matrix

    @staticmethod
    def guess_phase(k, characteristic_impedance, phase_velocity, capacitance, **_):
        return 2 * np.arctan(2 * np.pi * k * phase_velocity * capacitance * characteristic_impedance)


class PortMatrix:
    @staticmethod
    def scattering_matrix(**_):
        return np.array([[0]])
    
    @staticmethod
    def guess_phase(**_):
        return 0
    
if __name__ == '__main__':
    x = OpenMatrix()