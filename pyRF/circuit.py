from scipy.optimize import minimize
from scipy.linalg import null_space
import scipy.integrate
import scipy
import networkx as nx
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Circuit:
    def __init__(self, name):
        self.name = name
        self.circuit_elements: dict = None
        self.transmission_lines: dict = None

    def define_circuit_elements(self):
        pass

    def initialize_circuit_elements(self):
        return

    def define_transmission_lines(self):
        pass

    def initialize_transmission_lines(self):
        return
    
    def initialize(self):
        
        self.initialize_circuit_elements()
        self.initialize_transmission_lines()

    