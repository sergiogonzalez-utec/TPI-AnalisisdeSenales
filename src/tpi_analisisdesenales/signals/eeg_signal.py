"""
Modulo para senales EEG.
"""

from __future__ import annotations

import copy
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .raw_signal import RawSignal

from tpi_analisisdesenales.info import Info
from tpi_analisisdesenales.eventos import Anotaciones, Eventos
from tpi_analisisdesenales.epocas import Epocas

from tpi_analisisdesenales.preprocesamiento.filtros import (
    filtro_notch,
    filtro_pasabanda,
    filtro_pasabajos,
    filtro_pasaaltos,
    aplicar_hilbert,
    frecuencia_instantanea,
)


class EEGSignal(RawSignal):
    """
    Clase para representar senales EEG.

    Hereda de RawSignal y agrega metodos especificos para EEG:
    - filtrado EEG
    - notch
    - pasabanda
    - Hilbert
    - creacion de epocas
    - descripcion EEG
    """

    def __init__(
        self,
        data: np.ndarray,
        sfreq: float,
        ch_names: Sequence[str | int],
        ch_types: Optional[Sequence[str]] = None,
        times: Optional[np.ndarray] = None,
        montage: Optional[dict] = None,
        reference: Optional[str] = None,
        units: str = "uV",
        anotaciones: Optional[Anotaciones] = None,
        eventos: Optional[Eventos] = None,
        event_id: Optional[dict[str, int]] = None,
        subject_info: Optional[dict] = None,
        meas_date: Optional[str] = None,
        filters: Optional[list[dict]] = None,
        is_epoched: bool = False,
        is_filtered: bool = False,
        first_samp: int = 0,
    ) -> None:
        """
        Inicializa una senal EEG.
        """

        data = np.asarray(data)

        if data.dtype not in (np.float32, np.float64):
            data = data.astype(float)

        self._validate_data_dimensions(data)

        sfreq = float(sfreq)

        if sfreq <= 0:
            raise ValueError("sfreq debe ser mayor que cero.")

        ch_names = [str(ch) for ch in ch_names]

        if ch_types is None:
            ch_types = ["EEG"] * len(ch_names)
        else:
            ch_types = list(ch_types)

        self._validate_channel_info(
            data=data,
            ch_names=ch_names,
            ch_types=ch_types,
        )

        subject_info = subject_info if subject_info is not None else {}

        info = Info(
            ch_names=ch_names,
            sfreq=sfreq,
            ch_types=ch_types,
            subject_info=subject_info,
        )

        if eventos is None:
            eventos = Eventos(sfreq)

        if anotaciones is None:
            anotaciones = Anotaciones()

        super().__init__(
            data=data,
            info=info,
            eventos=eventos,
            anotaciones=anotaciones,
            first_samp=first_samp,
        )

        self.tipo_senal = "EEG"

        self.times = times
        self.montage = montage
        self.reference = reference
        self.units = units
        self.event_id = event_id if event_id is not None else {}
        self.subject_info = subject_info
        self.meas_date = meas_date
        self.filters = filters if filters is not None else []
        self.is_epoched = bool(is_epoched)
        self.is_filtered = bool(is_filtered)

        if self.times is None:
            self.times = np.arange(self.n_samples) / self.sfreq
        else:
            self.times = np.asarray(self.times)
            self._validate_times_length()
    @property
    def n_channels(self) -> int:
        """
        Devuelve la cantidad de canales.
        """

        if self.data.ndim == 2:
            return self.data.shape[0]

        return self.data.shape[1]

    @property
    def n_samples(self) -> int:
        """
        Devuelve la cantidad de muestras.
        """

        return self.data.shape[-1]

    @property
    def n_trials(self):
        """
        Devuelve la cantidad de trials si la senal es 3D.
        """

        if self.data.ndim == 3:
            return self.data.shape[0]

        return None

    @property
    def duration(self) -> float:
        """
        Devuelve la duracion en segundos.
        """

        return self.n_samples / self.sfreq

    def _validate_data_dimensions(self, data):
        """
        Valida dimensiones de data.
        """

        if data.ndim not in (2, 3):
            raise ValueError(
                "data debe tener forma n_canales x n_muestras "
                "o n_trials x n_canales x n_muestras."
            )

    def _validate_channel_info(self, data, ch_names, ch_types):
        """
        Valida nombres y tipos de canales.
        """

        if data.ndim == 2:
            n_channels = data.shape[0]
        else:
            n_channels = data.shape[1]

        if len(ch_names) != n_channels:
            raise ValueError("La cantidad de ch_names no coincide con n_canales.")

        if len(ch_types) != n_channels:
            raise ValueError("La cantidad de ch_types no coincide con n_canales.")

    def _validate_times_length(self):
        """
        Valida que times tenga el largo correcto.
        """

        if len(self.times) != self.n_samples:
            raise ValueError(
                "La longitud de times debe coincidir con n_muestras."
            )

    def copy(self):
        """
        Devuelve una copia independiente.
        """

        return copy.deepcopy(self)

    def _new_like(
        self,
        data,
        ch_names=None,
        ch_types=None,
        times=None,
        anotaciones=None,
        eventos=None,
        filters=None,
        is_filtered=None,
        is_epoched=None,
        first_samp=None,
        reference=None,
    ):
        """
        Crea una nueva instancia EEGSignal conservando metadata.
        """

        return EEGSignal(
            data=data,
            sfreq=self.sfreq,
            ch_names=ch_names if ch_names is not None else self.info.ch_names,
            ch_types=ch_types if ch_types is not None else self.info.ch_types,
            times=times,
            montage=self.montage,
            reference=reference if reference is not None else self.reference,
            units=self.units,
            anotaciones=anotaciones if anotaciones is not None else self.anotaciones,
            eventos=eventos if eventos is not None else self.eventos,
            event_id=self.event_id,
            subject_info=self.subject_info,
            meas_date=self.meas_date,
            filters=filters if filters is not None else self.filters,
            is_epoched=self.is_epoched if is_epoched is None else is_epoched,
            is_filtered=self.is_filtered if is_filtered is None else is_filtered,
            first_samp=self.first_samp if first_samp is None else first_samp,
        )

    def _resolve_picks(self, picks=None):
        """
        Convierte picks a indices de canales.
        """

        ch_names = self.info.ch_names

        if picks is None:
            return list(range(len(ch_names)))

        if isinstance(picks, (int, str)):
            picks = [picks]

        indices = []

        for pick in picks:
            if isinstance(pick, int):
                if pick < 0 or pick >= len(ch_names):
                    raise ValueError("Indice de canal fuera de rango.")

                indices.append(pick)

            elif isinstance(pick, str):
                if pick not in ch_names:
                    raise ValueError(f"Canal no encontrado: {pick}")

                indices.append(ch_names.index(pick))

            else:
                raise TypeError("picks debe contener indices o nombres de canales.")

        return indices

    def get_data(
        self,
        picks: Optional[Sequence[int | str] | int | str] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        n_samples: Optional[int] = None,
    ) -> np.ndarray:
        """
        Obtiene datos de la senal.
        """

        indices = self._resolve_picks(picks)

        if start is None:
            start = 0

        if n_samples is not None:
            stop = start + int(n_samples)

        if stop is None:
            stop = self.n_samples

        if start < 0 or stop > self.n_samples or stop <= start:
            raise ValueError("Intervalo de muestras invalido.")

        if self.data.ndim == 2:
            return self.data[indices, start:stop]

        return self.data[:, indices, start:stop]

    def get_channels(self, picks):
        """
        Devuelve una nueva EEGSignal con canales seleccionados.
        """

        indices = self._resolve_picks(picks)

        if self.data.ndim == 2:
            new_data = self.data[indices, :]
        else:
            new_data = self.data[:, indices, :]

        return self._new_like(
            data=new_data,
            ch_names=[self.info.ch_names[i] for i in indices],
            ch_types=[self.info.ch_types[i] for i in indices],
        )

    def crop(self, tmin: float = 0.0, tmax: Optional[float] = None):
        """
        Recorta la senal en segundos y devuelve una nueva EEGSignal.
        """

        if tmax is None:
            tmax = self.duration

        if tmin < 0 or tmax <= tmin or tmax > self.duration:
            raise ValueError("Intervalo temporal invalido.")

        start = int(round(tmin * self.sfreq))
        stop = int(round(tmax * self.sfreq))

        if self.data.ndim == 2:
            new_data = self.data[:, start:stop]
        else:
            new_data = self.data[:, :, start:stop]

        new_times = np.arange(new_data.shape[-1]) / self.sfreq

        return self._new_like(
            data=new_data,
            times=new_times,
            first_samp=self.first_samp + start,
        )

    def set_reference(self, ref_channel: str | int):
        """
        Cambia la referencia usando un canal.
        """

        ref_idx = self._resolve_picks(ref_channel)[0]

        if self.data.ndim == 2:
            ref_data = self.data[ref_idx:ref_idx + 1, :]
            new_data = self.data - ref_data
        else:
            ref_data = self.data[:, ref_idx:ref_idx + 1, :]
            new_data = self.data - ref_data

        return self._new_like(
            data=new_data,
            reference=self.info.ch_names[ref_idx],
        )

    def drop_channels(self, picks):
        """
        Elimina uno o mas canales.
        """

        drop_indices = set(self._resolve_picks(picks))

        keep_indices = [
            i for i in range(self.n_channels)
            if i not in drop_indices
        ]

        if len(keep_indices) == 0:
            raise ValueError("No puede descartarse todos los canales.")

        if self.data.ndim == 2:
            new_data = self.data[keep_indices, :]
        else:
            new_data = self.data[:, keep_indices, :]

        return self._new_like(
            data=new_data,
            ch_names=[self.info.ch_names[i] for i in keep_indices],
            ch_types=[self.info.ch_types[i] for i in keep_indices],
        )

    def filtrar(
        self,
        l_freq: float = 1.0,
        h_freq: float = 40.0,
        notch_freq: float = 50.0,
        q: float = 30.0,
        order: int = 4,
    ):
        """
        Aplica una cadena basica de filtrado EEG.

        Por defecto:
        - Notch 50 Hz
        - Pasabanda 1 a 40 Hz
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

        new_filters = list(self.filters)
        new_filters.append(
            {
                "type": "notch",
                "freq": notch_freq,
                "q": q,
            }
        )
        new_filters.append(
            {
                "type": "bandpass",
                "l_freq": l_freq,
                "h_freq": h_freq,
                "order": order,
            }
        )

        return self._new_like(
            data=data_filtrada,
            filters=new_filters,
            is_filtered=True,
        )

    def filter(self, l_freq: float, h_freq: float, order: int = 4):
        """
        Alias para aplicar filtro pasabanda.
        """

        new_data = filtro_pasabanda(
            data=self.data,
            sfreq=self.sfreq,
            l_freq=l_freq,
            h_freq=h_freq,
            order=order,
        )

        new_filters = list(self.filters)
        new_filters.append(
            {
                "type": "bandpass",
                "l_freq": l_freq,
                "h_freq": h_freq,
                "order": order,
            }
        )

        return self._new_like(
            data=new_data,
            filters=new_filters,
            is_filtered=True,
        )

    def notch_filter(self, freq: float = 50.0, q: float = 30.0):
        """
        Aplica filtro notch.
        """

        new_data = filtro_notch(
            data=self.data,
            sfreq=self.sfreq,
            freq=freq,
            q=q,
        )

        new_filters = list(self.filters)
        new_filters.append(
            {
                "type": "notch",
                "freq": freq,
                "q": q,
            }
        )

        return self._new_like(
            data=new_data,
            filters=new_filters,
            is_filtered=True,
        )

    def aplicar_hilbert(self):
        """
        Aplica transformada de Hilbert.
        """

        analytic_signal, envelope, phase = aplicar_hilbert(self.data)

        return analytic_signal, envelope, phase

    def frecuencia_instantanea(self):
        """
        Calcula frecuencia instantanea usando Hilbert.
        """

        _, _, phase = self.aplicar_hilbert()

        return frecuencia_instantanea(
            phase=phase,
            sfreq=self.sfreq,
        )

    def get_epochs(
        self,
        eventos: Optional[Eventos] = None,
        event_id: Optional[int | Sequence[int]] = None,
        tmin: float = -0.2,
        tmax: float = 0.8,
    ):
        """
        Crea epocas a partir de eventos.
        """

        if self.data.ndim == 3:
            raise ValueError(
                "La senal ya esta dividida en trials."
            )

        if eventos is None:
            eventos = self.eventos

        if eventos is None:
            raise ValueError("No hay eventos disponibles para crear epocas.")

        return Epocas(
            raw=self,
            eventos=eventos,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
        )

    def describe(self) -> pd.DataFrame:
        """
        Calcula estadisticas descriptivas por canal.
        """

        rows = []

        for i, ch_name in enumerate(self.info.ch_names):
            if self.data.ndim == 2:
                x = self.data[i, :]
            else:
                x = self.data[:, i, :].reshape(-1)

            rows.append(
                {
                    "name": ch_name,
                    "type": self.info.ch_types[i],
                    "mean": float(np.mean(x)),
                    "std": float(np.std(x)),
                    "min": float(np.min(x)),
                    "max": float(np.max(x)),
                    "ptp": float(np.ptp(x)),
                    "rms": float(np.sqrt(np.mean(x ** 2))),
                    "median": float(np.median(x)),
                }
            )

        return pd.DataFrame(rows)

    def describe_eeg(self):
        """
        Devuelve un resumen basico de EEG.
        """

        return {
            "tipo_senal": self.tipo_senal,
            "sfreq": self.sfreq,
            "n_canales": self.n_channels,
            "n_muestras": self.n_samples,
            "duration": self.duration,
            "is_filtered": self.is_filtered,
            "reference": self.reference,
            "units": self.units,
        }

    def plot(
        self,
        picks=None,
        start: float = 0.0,
        stop: Optional[float] = None,
    ):
        """
        Grafica la senal EEG continua usando plot_raw.
        """

        if self.data.ndim == 3:
            raise ValueError("plot() esta pensado para datos continuos 2D.")

        from tpi_analisisdesenales.visualizacion.plot_raw import plot_raw

        return plot_raw(
            raw=self,
            picks=picks,
            start=start,
            stop=stop,
            title="EEGSignal",
        )