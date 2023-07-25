from pyRF.resonator import Resonator


class FeedLine(Resonator):
    def __init__(self, name, number_of_channels) -> None:
        super().__init__(name, number_of_channels)