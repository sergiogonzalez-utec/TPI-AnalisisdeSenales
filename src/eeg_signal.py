from src.raw_signal import RawSignal

class EEGSignal(RawSignal):
    def __init__(self, data, sfreq, info, eventos=None, anotaciones=None):
        super().__init__(data, sfreq, info, eventos, anotaciones)
        # Podés agregar atributos específicos de EEG aquí