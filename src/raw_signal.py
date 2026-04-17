import numpy as np
from src.info import Info
from src.eventos import Eventos
from src.anotaciones import Anotaciones

class RawSignal:
    def __init__(self, data: np.ndarray, sfreq: float, info: Info, eventos=None, anotaciones=None, first_samp=0):
        """
        Clase RawSignal: representa una señal cruda con sus metadatos, eventos y anotaciones.

        Parameters
        ----------
        data : np.ndarray
            Matriz de datos (canales x muestras).
        sfreq : float
            Frecuencia de muestreo en Hz.
        info : Info
            Objeto Info con metadatos de la señal.
        eventos : Eventos, optional
            Objeto Eventos (si no se pasa, se crea uno vacío).
        anotaciones : Anotaciones, optional
            Objeto Anotaciones (si no se pasa, se crea uno vacío).
        first_samp : int, optional
            Índice de la primera muestra (por defecto 0).
        """
        self.data = data
        self.sfreq = sfreq
        self.info = info
        self.eventos = eventos if eventos else Eventos()
        self.anotaciones = anotaciones if anotaciones else Anotaciones()
        self.first_samp = first_samp

    def get_data(self):
        """Devuelve los datos crudos."""
        return self.data

    def describe(self):
        """Devuelve un resumen de la señal."""
        return {
            "channels": self.data.shape[0],
            "samples": self.data.shape[1],
            "sfreq": self.sfreq,
            "subject": self.info.subject_info
        }

    def drop_channels(self, ch_names: list):
        """Elimina canales de la señal según sus nombres."""
        indices = [self.info.ch_names.index(ch) for ch in ch_names if ch in self.info.ch_names]
        self.data = np.delete(self.data, indices, axis=0)
        self.info.ch_names = [ch for ch in self.info.ch_names if ch not in ch_names]

    def crop(self, start: int, end: int):
        """Recorta la señal entre dos índices de muestra."""
        self.data = self.data[:, start:end]

    def duration(self):
        """Devuelve la duración de la señal en segundos."""
        return self.data.shape[1] / self.sfreq

    def __str__(self):
        return f"RawSignal: {self.data.shape[0]} canales, {self.data.shape[1]} muestras, sfreq={self.sfreq} Hz"
