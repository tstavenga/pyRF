import numpy as np




class OpenMatrix:
    @staticmethod
    def scattering_matrix(k, position, direction_array):
        return np.array([[np.exp(-4j * np.pi * direction_array[0] * k * position)]])

    def guess_phase(self):
        return 0


class ShortMatrix:
    @staticmethod
    def scattering_matrix(k, position, direction_array):
        return np.array([[-np.exp(-4j * np.pi * direction_array[0] * k * position)]])


    def guess_phase(self):
        return np.pi
    
if __name__ == '__main__':
    x = OpenMatrix()
    bla = 8