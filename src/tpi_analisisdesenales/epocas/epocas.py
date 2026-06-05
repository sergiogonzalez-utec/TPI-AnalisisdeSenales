"""
Modulo para crear epocas a partir de eventos.
"""

import numpy as np


class Epocas:
    """
    Representa segmentos de senal extraidos alrededor de eventos.
    """

    def __init__(self, raw, eventos, tmin=-0.2, tmax=0.8):
        self.raw = raw
        self.eventos = eventos
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.sfreq = raw.sfreq
        self.ch_names = raw.info.ch_names

        self.data, self.times, self.metadata = self._crear_epocas()

    def _crear_epocas(self):
        eventos_df = self.eventos.get_events()

        if len(eventos_df) == 0:
            raise ValueError("No hay eventos para crear epocas.")

        inicio_offset = int(round(self.tmin * self.sfreq))
        fin_offset = int(round(self.tmax * self.sfreq))

        n_muestras_epoca = fin_offset - inicio_offset

        epocas = []
        metadata = []

        for _, evento in eventos_df.iterrows():
            sample_evento = int(evento["sample"])

            inicio = sample_evento + inicio_offset
            fin = sample_evento + fin_offset

            if inicio < 0:
                continue

            if fin > self.raw.data.shape[1]:
                continue

            epoca = self.raw.data[:, inicio:fin]

            if epoca.shape[1] == n_muestras_epoca:
                epocas.append(epoca)

                metadata.append(
                    {
                        "onset": float(evento["onset"]),
                        "sample": int(evento["sample"]),
                        "event_id": int(evento["event_id"]),
                        "description": str(evento["description"]),
                    }
                )

        if len(epocas) == 0:
            raise ValueError("No se pudo crear ninguna epoca valida.")

        data = np.stack(epocas, axis=0)
        times = np.arange(n_muestras_epoca) / self.sfreq + self.tmin

        return data, times, metadata

    def get_data(self):
        """
        Devuelve los datos de las epocas.

        Forma:
        n_epocas x n_canales x n_muestras
        """

        return self.data

    def average(self):
        """
        Calcula el promedio de las epocas.

        Devuelve:
        n_canales x n_muestras
        """

        return np.mean(self.data, axis=0)

    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        return (
            f"Epocas(n_epocas={len(self)}, "
            f"n_canales={self.data.shape[1]}, "
            f"n_muestras={self.data.shape[2]})"
        )