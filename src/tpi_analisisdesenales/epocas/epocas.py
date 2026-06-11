"""
Modulo para crear epocas a partir de eventos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class Epocas:
    """
    Representa segmentos de senal extraidos alrededor de eventos.

    La data queda con forma:
    n_epocas x n_canales x n_muestras
    """

    def __init__(
        self,
        raw,
        eventos=None,
        tmin: float = -0.2,
        tmax: float = 0.8,
        event_id=None,
        reject_out_of_bounds: bool = True,
    ):
        """
        Inicializa las epocas.

        Parameters
        ----------
        raw : RawSignal
            Senal continua.

        eventos : Eventos | None
            Contenedor de eventos. Si es None, usa raw.eventos.

        tmin : float
            Tiempo inicial de la epoca respecto al evento, en segundos.

        tmax : float
            Tiempo final de la epoca respecto al evento, en segundos.

        event_id : int | list[int] | None
            Codigo o codigos de evento a usar. Si es None, usa todos.

        reject_out_of_bounds : bool
            Si es True, descarta epocas que queden fuera de la senal.
            Si es False, lanza error cuando una epoca queda fuera.
        """

        self.raw = raw
        self.eventos = eventos if eventos is not None else raw.eventos
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.event_id = event_id
        self.reject_out_of_bounds = bool(reject_out_of_bounds)

        self.sfreq = float(raw.sfreq)
        self.ch_names = list(raw.info.ch_names)

        self._validar_parametros()

        self.data, self.times, self.metadata = self._crear_epocas()

    def _validar_parametros(self):
        """
        Valida parametros principales.
        """

        if self.eventos is None:
            raise ValueError("No hay eventos para crear epocas.")

        if self.sfreq <= 0:
            raise ValueError("sfreq debe ser mayor que cero.")

        if self.tmin >= self.tmax:
            raise ValueError("tmin debe ser menor que tmax.")

        if self.raw.data.ndim != 2:
            raise ValueError("raw.data debe tener forma n_canales x n_muestras.")

    def _filtrar_eventos(self, eventos_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra eventos segun event_id.
        """

        if self.event_id is None:
            return eventos_df

        if isinstance(self.event_id, int):
            ids = [self.event_id]
        else:
            ids = list(self.event_id)

        return eventos_df[eventos_df["event_id"].isin(ids)]

    def _crear_epocas(self):
        """
        Crea las epocas desde los eventos.
        """

        eventos_df = self.eventos.get_events()

        if len(eventos_df) == 0:
            raise ValueError("No hay eventos para crear epocas.")

        eventos_df = self._filtrar_eventos(eventos_df)

        if len(eventos_df) == 0:
            raise ValueError("No hay eventos que coincidan con event_id.")

        inicio_offset = int(round(self.tmin * self.sfreq))
        fin_offset = int(round(self.tmax * self.sfreq))

        n_muestras_epoca = fin_offset - inicio_offset

        if n_muestras_epoca <= 0:
            raise ValueError("La epoca debe tener al menos una muestra.")

        epocas = []
        metadata = []

        n_muestras_raw = self.raw.data.shape[1]

        for _, evento in eventos_df.iterrows():
            sample_evento = int(evento["sample"])

            inicio = sample_evento + inicio_offset
            fin = sample_evento + fin_offset

            fuera_de_limites = inicio < 0 or fin > n_muestras_raw

            if fuera_de_limites:
                if self.reject_out_of_bounds:
                    continue

                raise ValueError(
                    "Una epoca queda fuera de los limites de la senal."
                )

            epoca = self.raw.data[:, inicio:fin]

            if epoca.shape[1] != n_muestras_epoca:
                if self.reject_out_of_bounds:
                    continue

                raise ValueError("La epoca creada no tiene el largo esperado.")

            epocas.append(epoca)

            metadata.append(
                {
                    "onset": float(evento["onset"]),
                    "sample": int(evento["sample"]),
                    "event_id": int(evento["event_id"]),
                    "description": str(evento["description"]),
                    "start_sample": int(inicio),
                    "stop_sample": int(fin),
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

        Returns
        -------
        data : np.ndarray
            Arreglo con forma n_epocas x n_canales x n_muestras.
        """

        return self.data

    def get_metadata(self) -> pd.DataFrame:
        """
        Devuelve la metadata de las epocas como DataFrame.
        """

        return pd.DataFrame(self.metadata)

    def average(self):
        """
        Calcula el promedio de las epocas.

        Returns
        -------
        avg : np.ndarray
            Promedio con forma n_canales x n_muestras.
        """

        return np.mean(self.data, axis=0)

    def select_event_id(self, event_id):
        """
        Devuelve un nuevo objeto Epocas filtrado por event_id.
        """

        return Epocas(
            raw=self.raw,
            eventos=self.eventos,
            tmin=self.tmin,
            tmax=self.tmax,
            event_id=event_id,
            reject_out_of_bounds=self.reject_out_of_bounds,
        )

    def __len__(self):
        """
        Devuelve la cantidad de epocas.
        """

        return self.data.shape[0]

    def __getitem__(self, index):
        """
        Permite acceder a una epoca por indice.
        """

        return self.data[index]

    def __repr__(self):
        return (
            f"Epocas(n_epocas={len(self)}, "
            f"n_canales={self.data.shape[1]}, "
            f"n_muestras={self.data.shape[2]}, "
            f"tmin={self.tmin}, "
            f"tmax={self.tmax})"
        )