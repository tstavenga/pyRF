from pyRF.core.circuit import Circuit

PHASE_VELOCITY = 124998795.11524159
LENGTH = 8.095835673090729e-3

class HalfWaveResonator(Circuit):
    def __init__(self, 
                 length: float, 
                 phase_velocity: float,
                 capacitance_start: float = 0, 
                 capacitance_end: float = 0,
                 characteristic_impedance: float = 50.0):
        super().__init__()
        self.phase_velocity = phase_velocity
        self.length = length
        self.capacitance_start = capacitance_start
        self.capacitance_end = capacitance_end
        self.characteristic_impedance = characteristic_impedance
        self.initialized = False

    def define_circuit_elements(self):

        self.circuit_elements = {
            'C1': {
                'element': 'Capacitor',
                'values': {
                    'capacitance':self.capacitance_start,
                    'position': 0,
                }
            },

            'C2': {'element': 'Capacitor',
                   'values': {
                       'capacitance':self.capacitance_end,                       
                       'position': self.length,
                       }
                    },
        }


    def define_resonators(self):
        self.resonators = {
            'R1': {
                'Capacitor_Short': {
                    'start_pin': {
                        'element': 'C1',
                        'side':'main',
                        'pin': 'alpha',
                    },
                    'end_pin': {
                        'element': 'C2',
                        'side':'main',
                        'pin': 'alpha',
                    },
                    'transmission_line': {
                        'characteristic_impedance': self.characteristic_impedance,
                        'phase_velocity': self.phase_velocity,
                    },
                }
            }
        }

    def get_frequency(self):
        if not self.initialized:
            self.initialize()
        
        resonator = self.resonator_dict['R1']
        return resonator.get_eigenvalue()


        


if __name__ == '__main__':

    phase_velocity = 120e6
    length = 8e-3
    half_wave_circuit = HalfWave(phase_velocity=phase_velocity,
                                    length=length,
                                    capacitance_start=30e-15,
                                    capacitance_end=30e-15)
    half_wave_circuit.initialize()
    R1 = half_wave_circuit.resonator_dict['R1']
    eigv = R1.get_eigenvalue()
    print(half_wave_circuit.get_frequency())
    print(eigv)
