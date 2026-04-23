from src.raw_signal import RawSignal

class EMGSignal(RawSignal):
    def __init__(self, data, sfreq, info, eventos=None, anotaciones=None):
        super().__init__(data, sfreq, info, eventos, anotaciones)
        # Aquí podés agregar atributos o métodos específicos de EMG
        # Por ejemplo, normalización de amplitud o detección de contracciones
