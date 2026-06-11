"""
Modulo para senales ECG.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from .raw_signal import RawSignal
from tpi_analisisdesenales.preprocesamiento.filtros import (
    filtro_notch,
    filtro_pasaaltos,
    filtro_pasabajos,
)


class ECGSignal(RawSignal):
    """
    Representa una senal ECG.

    Hereda de RawSignal y agrega metodos especificos:
    - filtrado ECG
    - deteccion de picos R
    - calculo de frecuencia cardiaca promedio
    """

    def __init__(self, data, info, anotaciones=None, eventos=None):
        super().__init__(
            data=data,
            info=info,
            anotaciones=anotaciones,
            eventos=eventos,
        )

        self.tipo_senal = "ECG"
        self.r_peaks = None
        self.heart_rate = None

    def filtrar(
        self,
        l_freq: float = 0.5,
        h_freq: float = 40.0,
        notch_freq: float = 50.0,
        q: float = 30.0,
    ):
        """
        Aplica una cadena basica de filtrado ECG.

        Returns
        -------
        ecg_filtrado : ECGSignal
            Nueva senal ECG filtrada.
        """

        data_filtrada = filtro_notch(
            self.data,
            self.sfreq,
            freq=notch_freq,
            q=q,
        )

        data_filtrada = filtro_pasaaltos(
            data_filtrada,
            self.sfreq,
            l_freq=l_freq,
            order=4,
        )

        data_filtrada = filtro_pasabajos(
            data_filtrada,
            self.sfreq,
            h_freq=h_freq,
            order=4,
        )

        return ECGSignal(
            data=data_filtrada,
            info=self.info,
            anotaciones=self.anotaciones,
            eventos=self.eventos,
        )

    def detectar_picos_r(
        self,
        canal: int = 0,
        distancia_minima_s: float = 0.3,
    ):
        """
        Detecta picos R de forma simple usando scipy.signal.find_peaks.

        Parameters
        ----------
        canal : int
            Canal ECG a analizar.

        distancia_minima_s : float
            Distancia minima entre picos en segundos.

        Returns
        -------
        r_peaks : np.ndarray
            Indices de muestra de los picos R.
        """

        if canal < 0 or canal >= self.data.shape[0]:
            raise ValueError("canal fuera de rango.")

        senal = self.data[canal]

        distancia = int(round(distancia_minima_s * self.sfreq))

        altura_minima = np.mean(senal) + 0.5 * np.std(senal)

        peaks, _ = find_peaks(
            senal,
            distance=distancia,
            height=altura_minima,
        )

        self.r_peaks = peaks

        return peaks

    def calcular_frecuencia_cardiaca(self, canal: int = 0):
        """
        Calcula la frecuencia cardiaca promedio en BPM.

        Returns
        -------
        heart_rate : float
            Frecuencia cardiaca promedio.
        """

        if self.r_peaks is None:
            self.detectar_picos_r(canal=canal)

        if len(self.r_peaks) < 2:
            raise ValueError(
                "No hay suficientes picos R para calcular frecuencia cardiaca."
            )

        rr = np.diff(self.r_peaks) / self.sfreq

        self.heart_rate = 60.0 / np.mean(rr)

        return self.heart_rate

    def describe_ecg(self, canal: int = 0):
        """
        Devuelve un resumen basico del ECG.
        """

        r_peaks = self.detectar_picos_r(canal=canal)
        heart_rate = self.calcular_frecuencia_cardiaca(canal=canal)

        return {
            "tipo_senal": self.tipo_senal,
            "canal": canal,
            "n_picos_r": len(r_peaks),
            "heart_rate_bpm": heart_rate,
            "sfreq": self.sfreq,
            "n_canales": self.data.shape[0],
            "n_muestras": self.data.shape[1],
        }