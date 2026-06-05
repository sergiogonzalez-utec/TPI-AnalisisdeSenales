"""
Modulo para manejar anotaciones temporales en senales biologicas.

Una anotacion representa una marca temporal sobre una senal continua.
Puede ser puntual, si duration = 0, o puede representar un intervalo,
si duration > 0.
"""

from __future__ import annotations

import pandas as pd


class Anotaciones:
    """
    Clase para almacenar anotaciones asociadas a una senal.

    Cada anotacion tiene:
    - onset: inicio en segundos
    - duration: duracion en segundos
    - description: descripcion textual
    """

    def __init__(self, anotaciones=None):
        """
        Inicializa el contenedor de anotaciones.

        Parameters
        ----------
        anotaciones : list[dict] | None
            Lista inicial de anotaciones.
        """

        if anotaciones is None:
            self._anotaciones = []
        else:
            self._anotaciones = list(anotaciones)

    def add_annotation(
        self,
        onset: float,
        duration: float = 0.0,
        description: str = "Anotacion",
    ):
        """
        Agrega una anotacion.

        Parameters
        ----------
        onset : float
            Tiempo inicial en segundos.

        duration : float
            Duracion en segundos.

        description : str
            Descripcion de la anotacion.
        """

        onset = float(onset)
        duration = float(duration)

        if onset < 0:
            raise ValueError("onset no puede ser negativo.")

        if duration < 0:
            raise ValueError("duration no puede ser negativo.")

        self._anotaciones.append(
            {
                "onset": onset,
                "duration": duration,
                "description": str(description),
            }
        )

    def get_annotations(self) -> pd.DataFrame:
        """
        Devuelve las anotaciones como DataFrame.
        """

        return pd.DataFrame(
            self._anotaciones,
            columns=["onset", "duration", "description"],
        )

    def to_list(self):
        """
        Devuelve las anotaciones como lista de diccionarios.
        """

        return list(self._anotaciones)

    def clear(self):
        """
        Elimina todas las anotaciones.
        """

        self._anotaciones.clear()

    def __iter__(self):
        """
        Permite iterar directamente sobre las anotaciones.
        """

        return iter(self._anotaciones)

    def __len__(self):
        """
        Devuelve la cantidad de anotaciones.
        """

        return len(self._anotaciones)

    def __repr__(self):
        return f"Anotaciones(n={len(self)})"