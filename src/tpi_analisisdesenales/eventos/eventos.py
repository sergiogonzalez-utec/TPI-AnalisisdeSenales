"""
Modulo para manejar eventos puntuales en senales biologicas.
"""

from __future__ import annotations

import pandas as pd


class Eventos:
    """
    Representa eventos puntuales asociados a una senal.

    Cada evento tiene:
    - onset: tiempo del evento en segundos
    - sample: muestra correspondiente
    - event_id: codigo numerico del evento
    - description: descripcion textual
    """

    def __init__(self, sfreq: float, eventos=None):
        """
        Inicializa el contenedor de eventos.

        Parameters
        ----------
        sfreq : float
            Frecuencia de muestreo en Hz.

        eventos : list[dict] | None
            Lista inicial de eventos.
        """

        self.sfreq = float(sfreq)

        if self.sfreq <= 0:
            raise ValueError("sfreq debe ser mayor que cero.")

        if eventos is None:
            self._eventos = []
        else:
            self._eventos = list(eventos)

    def add_event(
        self,
        onset: float,
        event_id: int,
        description: str = "Evento",
    ):
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

        onset = float(onset)

        if onset < 0:
            raise ValueError("onset no puede ser negativo.")

        sample = int(round(onset * self.sfreq))

        self._eventos.append(
            {
                "onset": onset,
                "sample": sample,
                "event_id": int(event_id),
                "description": str(description),
            }
        )

    def add_event_sample(
        self,
        sample: int,
        event_id: int,
        description: str = "Evento",
    ):
        """
        Agrega un evento a partir del numero de muestra.

        Parameters
        ----------
        sample : int
            Indice de muestra donde ocurre el evento.

        event_id : int
            Codigo numerico del evento.

        description : str
            Descripcion del evento.
        """

        sample = int(sample)

        if sample < 0:
            raise ValueError("sample no puede ser negativo.")

        onset = sample / self.sfreq

        self._eventos.append(
            {
                "onset": float(onset),
                "sample": sample,
                "event_id": int(event_id),
                "description": str(description),
            }
        )

    def rename_event_id(self, old_id: int, new_id: int):
        """
        Renombra un codigo de evento.

        Parameters
        ----------
        old_id : int
            Codigo anterior.

        new_id : int
            Codigo nuevo.
        """

        old_id = int(old_id)
        new_id = int(new_id)

        for evento in self._eventos:
            if evento["event_id"] == old_id:
                evento["event_id"] = new_id

    def find(self, text: str):
        """
        Busca eventos cuya descripcion contenga el texto indicado.
        """

        text = str(text).lower()

        return [
            evento
            for evento in self._eventos
            if text in evento["description"].lower()
        ]

    def get_events(self) -> pd.DataFrame:
        """
        Devuelve los eventos como DataFrame.
        """

        return pd.DataFrame(
            self._eventos,
            columns=["onset", "sample", "event_id", "description"],
        )

    def to_list(self):
        """
        Devuelve los eventos como lista de diccionarios.
        """

        return list(self._eventos)

    def clear(self):
        """
        Elimina todos los eventos.
        """

        self._eventos.clear()

    def __iter__(self):
        """
        Permite iterar directamente sobre los eventos.
        """

        return iter(self._eventos)

    def __getitem__(self, index):
        """
        Permite acceder a un evento por indice.
        """

        return self._eventos[index]

    def __len__(self):
        """
        Devuelve la cantidad de eventos.
        """

        return len(self._eventos)

    def __repr__(self):
        return f"Eventos(n={len(self)}, sfreq={self.sfreq})"