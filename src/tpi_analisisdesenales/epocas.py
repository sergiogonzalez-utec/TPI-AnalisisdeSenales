import numpy as np
from tpi_analisisdesenales.eventos import Eventos
from tpi_analisisdesenales.raw_signal import RawSignal

class Epocas:
    def __init__(self, signal: RawSignal, eventos: Eventos, tmin: float, tmax: float, picks=None, reject=None):
        """
        Clase Epocas: representa segmentos de la señal (epochs) definidos por eventos.

        Parameters
        ----------
        signal : RawSignal
            Señal cruda de la cual se extraen las épocas.
        eventos : Eventos
            Objeto Eventos que contiene los marcadores.
        tmin : float
            Tiempo inicial relativo al evento (en segundos).
        tmax : float
            Tiempo final relativo al evento (en segundos).
        picks : list, optional
            Lista de canales a incluir.
        reject : dict, optional
            Criterios de rechazo (ej. amplitud máxima).
        """
        self.signal = signal
        self.eventos = eventos
        self.tmin = tmin
        self.tmax = tmax
        self.picks = picks or signal.info.ch_names
        self.reject = reject or {}
        self.data = self._extract_epochs()

    def _extract_epochs(self):
        """Extrae las épocas de la señal según los eventos."""
        epochs = []
        samples_tmin = int(self.tmin * self.signal.sfreq)
        samples_tmax = int(self.tmax * self.signal.sfreq)

        for ev in self.eventos.get_events():
            start = ev["sample"] + samples_tmin
            end = ev["sample"] + samples_tmax
            if 0 <= start < self.signal.data.shape[1] and end <= self.signal.data.shape[1]:
                epochs.append(self.signal.data[:, start:end])
        return np.array(epochs)

    def get_data(self):
        """Devuelve los datos de las épocas."""
        return self.data

    def average(self):
        """Calcula el promedio de todas las épocas."""
        return np.mean(self.data, axis=0) if len(self.data) > 0 else None

    def drop_channels(self, ch_names: list):
        """Elimina canales de las épocas."""
        indices = [self.signal.info.ch_names.index(ch) for ch in ch_names if ch in self.signal.info.ch_names]
        self.data = np.delete(self.data, indices, axis=1)
        self.signal.info.ch_names = [ch for ch in self.signal.info.ch_names if ch not in ch_names]

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return f"Epocas: {len(self)} segmentos de {self.tmin}-{self.tmax} s"
