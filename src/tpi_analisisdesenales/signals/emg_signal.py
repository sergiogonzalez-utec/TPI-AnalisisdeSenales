"""
Modulo para senales EMG.
"""

from __future__ import annotations

import numpy as np

from .raw_signal import RawSignal

from tpi_analisisdesenales.eventos.anotaciones import Anotaciones
from tpi_analisisdesenales.eventos.eventos import Eventos
from tpi_analisisdesenales.info.info import Info

from tpi_analisisdesenales.preprocesamiento.filtros import (
    filtro_notch,
    filtro_pasaaltos,
    filtro_pasabajos,
    filtro_pasabanda,
)


class EMGSignal(RawSignal):
    """
    Clase para representar y analizar senales EMG.

    Hereda la funcionalidad general de RawSignal y agrega
    metodos especificos para analisis basico de electromiografia.
    """

    def __init__(
        self,
        data: np.ndarray,
        info: Info,
        eventos: Eventos | None = None,
        anotaciones: Anotaciones | None = None,
        first_samp: int = 0,
        muscle_group: str | None = None,
        sampling_window: int | None = None,
        filters_applied: list | None = None,
        features: dict | None = None,
    ):
        """
        Inicializa una instancia de EMGSignal.

        Parameters
        ----------
        data : np.ndarray
            Matriz con forma n_canales x n_muestras.

        info : Info
            Objeto con metadata de la senal.

        eventos : Eventos | None
            Eventos discretos asociados.

        anotaciones : Anotaciones | None
            Anotaciones temporales asociadas.

        first_samp : int
            Indice de la primera muestra valida.

        muscle_group : str | None
            Nombre general del grupo muscular registrado.

        sampling_window : int | None
            Tamano de ventana sugerido para procesamiento.

        filters_applied : list | None
            Lista con filtros aplicados a la senal.

        features : dict | None
            Diccionario para guardar caracteristicas extraidas.
        """

        super().__init__(
            data=data,
            info=info,
            eventos=eventos,
            anotaciones=anotaciones,
            first_samp=first_samp,
        )

        self.tipo_senal = "EMG"
        self.muscle_group = muscle_group
        self.sampling_window = sampling_window
        self.filters_applied = filters_applied if filters_applied is not None else []
        self.features = features if features is not None else {}

        self.validate_structure()

    def validate_structure(self):
        """
        Valida la estructura basica de la senal EMG.
        """

        if self.data.ndim != 2:
            raise ValueError("La senal EMG debe ser una matriz 2D.")

        if self.data.shape[0] < 1:
            raise ValueError("La senal EMG debe tener al menos un canal.")

        if self.data.shape[1] < 1:
            raise ValueError("La senal EMG debe tener al menos una muestra.")

        if len(self.info.ch_names) != self.data.shape[0]:
            raise ValueError(
                "La cantidad de nombres de canal no coincide con la senal."
            )

        if len(self.info.ch_types) != self.data.shape[0]:
            raise ValueError(
                "La cantidad de tipos de canal no coincide con la senal."
            )

        return True

    def filtrar(
        self,
        l_freq: float = 20.0,
        h_freq: float = 150.0,
        notch_freq: float = 50.0,
        q: float = 30.0,
        order: int = 4,
    ):
        """
        Aplica una cadena basica de filtrado EMG.

        Por defecto:
        - Notch 50 Hz
        - Pasabanda 20 a 150 Hz

        Returns
        -------
        emg_filtrado : EMGSignal
            Nueva senal EMG filtrada.
        """

        data_filtrada = filtro_notch(
            self.data,
            self.sfreq,
            freq=notch_freq,
            q=q,
        )

        data_filtrada = filtro_pasabanda(
            data_filtrada,
            self.sfreq,
            l_freq=l_freq,
            h_freq=h_freq,
            order=order,
        )

        filters_applied = list(self.filters_applied)
        filters_applied.append(
            {
                "type": "notch",
                "freq": notch_freq,
                "q": q,
            }
        )
        filters_applied.append(
            {
                "type": "bandpass",
                "l_freq": l_freq,
                "h_freq": h_freq,
                "order": order,
            }
        )

        return EMGSignal(
            data=data_filtrada,
            info=self.info,
            eventos=self.eventos,
            anotaciones=self.anotaciones,
            first_samp=self.first_samp,
            muscle_group=self.muscle_group,
            sampling_window=self.sampling_window,
            filters_applied=filters_applied,
            features=self.features,
        )

    def rectificar(self):
        """
        Rectifica la senal EMG.

        Returns
        -------
        rectificada : np.ndarray
            Senal rectificada.
        """

        return np.abs(self.data)

    def envelope(self, rectify: bool = True):
        """
        Calcula una envolvente basica de la senal EMG.

        En esta version, la envolvente se calcula como valor absoluto
        si rectify=True.
        """

        if rectify:
            return np.abs(self.data)

        return self.data.copy()

    def rms_per_channel(self):
        """
        Calcula el RMS por canal.

        Returns
        -------
        rms : np.ndarray
            Vector con un valor RMS por canal.
        """

        return np.sqrt(np.mean(self.data ** 2, axis=1))

    def mav_per_channel(self):
        """
        Calcula el valor medio absoluto por canal.

        Returns
        -------
        mav : np.ndarray
            Vector con un valor MAV por canal.
        """

        return np.mean(np.abs(self.data), axis=1)

    def variance_per_channel(self):
        """
        Calcula la varianza por canal.

        Returns
        -------
        var : np.ndarray
            Vector con una varianza por canal.
        """

        return np.var(self.data, axis=1)

    def describe(self):
        """
        Devuelve estadisticas descriptivas por canal para EMG.

        Extiende describe() de RawSignal agregando columnas EMG.
        """

        df = super().describe()

        df["emg_rms"] = self.rms_per_channel()
        df["emg_mav"] = self.mav_per_channel()
        df["emg_var"] = self.variance_per_channel()

        return df

    def feature_extraction(self):
        """
        Extrae caracteristicas basicas EMG y las guarda.

        Returns
        -------
        features : dict
            Diccionario con rms, mav y var.
        """

        self.features = {
            "rms": self.rms_per_channel(),
            "mav": self.mav_per_channel(),
            "var": self.variance_per_channel(),
        }

        return self.features

    def describe_emg(self):
        """
        Devuelve un resumen basico de la senal EMG.
        """

        features = self.feature_extraction()

        return {
            "tipo_senal": self.tipo_senal,
            "sfreq": self.sfreq,
            "n_canales": self.data.shape[0],
            "n_muestras": self.data.shape[1],
            "muscle_group": self.muscle_group,
            "rms": features["rms"],
            "mav": features["mav"],
            "var": features["var"],
        }