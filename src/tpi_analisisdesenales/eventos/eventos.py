"""
Modulo para manejar eventos puntuales en senales biologicas.
"""

import pandas as pd


class Eventos:
    """
    Representa eventos puntuales asociados a una senal.
    """

    def __init__(self, sfreq):
        self.sfreq = float(sfreq)
        self._eventos = []

    def add_event(self, onset, event_id, description="Evento"):
        """
        Agrega un evento puntual.

        Parameters
        ----------
        onset : float
            Tiempo del evento en segundos.

        event_id : int
            Codigo numerico del evento.

        description : str
            Descripcion del evento.
        """

        if onset < 0:
            raise ValueError("onset no puede ser negativo.")

        sample = int(round(onset * self.sfreq))

        self._eventos.append(
            {
                "onset": float(onset),
                "sample": sample,
                "event_id": int(event_id),
                "description": str(description),
            }
        )

    def get_events(self):
        """
        Devuelve los eventos como DataFrame.
        """

        return pd.DataFrame(
            self._eventos,
            columns=["onset", "sample", "event_id", "description"],
        )

    def __len__(self):
        return len(self._eventos)

    def __repr__(self):
        return f"Eventos(n={len(self)}, sfreq={self.sfreq})"