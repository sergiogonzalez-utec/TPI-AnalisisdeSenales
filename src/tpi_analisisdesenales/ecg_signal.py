from tpi_analisisdesenales.raw_signal import RawSignal

class ECGSignal(RawSignal):
    def __init__(self, data, sfreq, info, eventos=None, anotaciones=None):
        super().__init__(data, sfreq, info, eventos, anotaciones)
        # Aquí podés agregar atributos o métodos específicos de ECG