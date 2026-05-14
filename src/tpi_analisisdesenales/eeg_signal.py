from __future__ import annotations

import copy
import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .info import Info
from .anotaciones import Anotaciones
from .eventos import Eventos


class EEGSignal:
    """
    Clase para representar senales de EEG.

    Permite trabajar con datos continuos de forma:
    n_canales x n_muestras

    o con datos divididos en trials de forma:
    n_trials x n_canales x n_muestras
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

        self.data = np.asarray(data)

        if self.data.dtype not in (np.float32, np.float64):
            self.data = self.data.astype(float)

        self.sfreq = float(sfreq)
        self.ch_names = [str(ch) for ch in ch_names]

        if ch_types is None:
            self.ch_types = ["EEG"] * len(self.ch_names)
        else:
            self.ch_types = list(ch_types)

        self.times = times
        self.montage = montage
        self.reference = reference
        self.units = units

        self.anotaciones = anotaciones if anotaciones is not None else Anotaciones()
        self.eventos = eventos
        self.event_id = event_id if event_id is not None else {}

        self.subject_info = subject_info if subject_info is not None else {}
        self.meas_date = meas_date
        self.filters = filters if filters is not None else []

        self.is_epoched = bool(is_epoched)
        self.is_filtered = bool(is_filtered)
        self.first_samp = int(first_samp)

        self._validate_dimensions()
        self._validate_channel_info()

        if self.times is None:
            self.times = np.arange(self.n_samples) / self.sfreq
        else:
            self.times = np.asarray(self.times)
            self._validate_times_length()
            self._validate_time_sfreq_consistency()

        self.info = Info(
            ch_names=self.ch_names,
            sfreq=self.sfreq,
            ch_types=self.ch_types,
        )

    @property
    def n_channels(self) -> int:
        if self.data.ndim == 2:
            return self.data.shape[0]
        return self.data.shape[1]

    @property
    def n_samples(self) -> int:
        return self.data.shape[-1]

    @property
    def n_trials(self) -> Optional[int]:
        if self.data.ndim == 3:
            return self.data.shape[0]
        return None

    @property
    def duration(self) -> float:
        return self.n_samples / self.sfreq

    def _validate_dimensions(self) -> None:
        """
        Valida que los datos sean 2D o 3D.
        """

        if self.data.ndim not in (2, 3):
            raise ValueError(
                "data debe tener forma n_canales x n_muestras "
                "o n_trials x n_canales x n_muestras."
            )

    def _validate_channel_info(self) -> None:
        """
        Valida que la cantidad de nombres y tipos de canales coincida
        con la cantidad de canales de la senal.
        """

        if len(self.ch_names) != self.n_channels:
            raise ValueError("La cantidad de ch_names no coincide con n_canales.")

        if len(self.ch_types) != self.n_channels:
            raise ValueError("La cantidad de ch_types no coincide con n_canales.")

    def _validate_times_length(self) -> None:
        """
        Valida que el vector temporal tenga la misma cantidad de muestras.
        """

        if len(self.times) != self.n_samples:
            raise ValueError(
                "La longitud del vector times debe coincidir con n_muestras."
            )

    def _validate_time_sfreq_consistency(self) -> None:
        """
        Valida que el paso temporal coincida con la frecuencia de muestreo.
        """

        if len(self.times) < 2:
            return

        dt = np.diff(self.times)
        expected_dt = 1.0 / self.sfreq

        if not np.allclose(dt, expected_dt, rtol=1e-3, atol=1e-6):
            warnings.warn(
                "El vector times no es consistente con la frecuencia de muestreo.",
                UserWarning,
            )

    def copy(self) -> "EEGSignal":
        """
        Devuelve una copia independiente del objeto.
        """

        return copy.deepcopy(self)

    def set_annotations(self, anotaciones: Anotaciones) -> None:
        """
        Asigna anotaciones a la senal EEG.
        """

        if not isinstance(anotaciones, Anotaciones):
            raise TypeError("anotaciones debe ser una instancia de Anotaciones.")

        self.anotaciones = anotaciones

    def get_data(
        self,
        picks: Optional[Sequence[int | str] | int | str] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        n_samples: Optional[int] = None,
    ) -> np.ndarray:
        """
        Obtiene datos de la senal.

        Returns
        -------
        data : np.ndarray
            Si la senal es continua:
            n_canales x n_muestras

            Si la senal tiene trials:
            n_trials x n_canales x n_muestras
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

    def get_channels(
        self,
        picks: Sequence[int | str] | int | str,
    ) -> "EEGSignal":
        """
        Obtiene uno o mas canales y retorna una nueva instancia de EEGSignal.
        """

        indices = self._resolve_picks(picks)

        if self.data.ndim == 2:
            new_data = self.data[indices, :]
        else:
            new_data = self.data[:, indices, :]

        return self._new_like(
            data=new_data,
            ch_names=[self.ch_names[i] for i in indices],
            ch_types=[self.ch_types[i] for i in indices],
        )

    def crop(self, tmin: float = 0.0, tmax: Optional[float] = None) -> "EEGSignal":
        """
        Recorta temporalmente la senal y retorna una nueva instancia de EEGSignal.
        """

        if tmax is None:
            tmax = self.duration

        if tmin < 0 or tmax <= tmin or tmax > self.duration:
            raise ValueError("Intervalo temporal invalido.")

        start = int(np.floor(tmin * self.sfreq))
        stop = int(np.ceil(tmax * self.sfreq))

        if self.data.ndim == 2:
            new_data = self.data[:, start:stop]
        else:
            new_data = self.data[:, :, start:stop]

        new_times = self.times[start:stop] - tmin

        new_annotations = self._crop_annotations(tmin=tmin, tmax=tmax)

        return self._new_like(
            data=new_data,
            times=new_times,
            anotaciones=new_annotations,
            first_samp=self.first_samp + start,
        )

    def set_reference(self, ref_channel: str | int) -> "EEGSignal":
        """
        Cambia la referencia de la senal a un canal especifico.

        Retorna una nueva instancia de EEGSignal.
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
            reference=self.ch_names[ref_idx],
        )

    def drop_channels(
        self,
        picks: Sequence[int | str] | int | str,
    ) -> "EEGSignal":
        """
        Descarta uno o mas canales y retorna una nueva instancia de EEGSignal.
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
            ch_names=[self.ch_names[i] for i in keep_indices],
            ch_types=[self.ch_types[i] for i in keep_indices],
        )

    def describe(self) -> pd.DataFrame:
        """
        Calcula caracteristicas descriptivas por canal.
        """

        rows = []

        for i, ch_name in enumerate(self.ch_names):
            if self.data.ndim == 2:
                x = self.data[i, :]
            else:
                x = self.data[:, i, :].reshape(-1)

            rows.append(
                {
                    "channel": ch_name,
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

    def filter(self, l_freq: float, h_freq: float, order: int = 4) -> "EEGSignal":
        """
        Aplica un filtro pasabanda y retorna una nueva instancia de EEGSignal.
        """

        from .preprocesamiento.filtros import filtro_pasabanda

        new_data = filtro_pasabanda(
            data=self.data,
            sfreq=self.sfreq,
            l_freq=l_freq,
            h_freq=h_freq,
            order=order,
        )

        new_filters = self.filters + [
            {
                "type": "bandpass",
                "l_freq": l_freq,
                "h_freq": h_freq,
                "order": order,
            }
        ]

        return self._new_like(
            data=new_data,
            filters=new_filters,
            is_filtered=True,
        )

    def notch_filter(self, freq: float = 50.0, q: float = 30.0) -> "EEGSignal":
        """
        Aplica un filtro Notch y retorna una nueva instancia de EEGSignal.
        """

        from .preprocesamiento.filtros import filtro_notch

        new_data = filtro_notch(
            data=self.data,
            sfreq=self.sfreq,
            freq=freq,
            q=q,
        )

        new_filters = self.filters + [
            {
                "type": "notch",
                "freq": freq,
                "q": q,
            }
        ]

        return self._new_like(
            data=new_data,
            filters=new_filters,
            is_filtered=True,
        )

    def get_epochs(
        self,
        eventos: Optional[Eventos] = None,
        event_id: Optional[int | Sequence[int]] = None,
        tmin: float = -0.2,
        tmax: float = 0.8,
    ):
        """
        Obtiene epocas a partir de eventos.

        Retorna una instancia de Epocas.
        """

        from .epocas import Epocas

        if self.data.ndim == 3:
            raise ValueError(
                "La senal ya esta dividida en trials. "
                "Use los datos actuales o conviertalos segun corresponda."
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

    def plot(self, picks=None, start: float = 0.0, stop: Optional[float] = None):
        """
        Grafica la senal EEG continua usando el modulo de visualizacion.
        """

        if self.data.ndim == 3:
            raise ValueError("plot() esta pensado para datos continuos 2D.")

        from .visualizacion.plot_raw import plot_raw

        return plot_raw(
            raw=self,
            picks=picks,
            start=start,
            stop=stop,
            title="EEGSignal",
        )

    def plot_spectrum(self, picks=None):
        """
        Calcula y grafica el espectro de Fourier.
        """

        import matplotlib.pyplot as plt

        indices = self._resolve_picks(picks)
        fig, ax = plt.subplots(figsize=(10, 5))

        freqs = np.fft.rfftfreq(self.n_samples, d=1.0 / self.sfreq)

        for idx in indices:
            if self.data.ndim == 2:
                x = self.data[idx, :]
                spectrum = np.abs(np.fft.rfft(x))
            else:
                x = self.data[:, idx, :]
                spectrum = np.mean(np.abs(np.fft.rfft(x, axis=-1)), axis=0)

            ax.plot(freqs, spectrum, label=self.ch_names[idx])

        ax.set_title("Espectro de Fourier")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Amplitud")
        ax.grid(True)
        ax.legend()

        return fig

    def plot_time_frequency(
        self,
        channel: int | str,
        nperseg: int = 128,
        noverlap: Optional[int] = None,
        trial: int = 0,
    ):
        """
        Calcula y grafica una representacion tiempo-frecuencia para un canal.
        """

        import matplotlib.pyplot as plt
        from scipy.signal import spectrogram

        idx = self._resolve_picks(channel)[0]

        if self.data.ndim == 2:
            x = self.data[idx, :]
        else:
            x = self.data[trial, idx, :]

        f, t, sxx = spectrogram(
            x,
            fs=self.sfreq,
            nperseg=nperseg,
            noverlap=noverlap,
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        mesh = ax.pcolormesh(t, f, sxx, shading="auto")
        fig.colorbar(mesh, ax=ax, label="Potencia")

        ax.set_title(f"Tiempo-frecuencia - {self.ch_names[idx]}")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Frecuencia (Hz)")

        return fig

    def hilbert_transform(self, picks=None):
        """
        Calcula la transformada de Hilbert para uno o mas canales.

        El calculo real se delega al modulo preprocesamiento/hilbert.py.
        """

        from .preprocesamiento.hilbert import aplicar_hilbert

        indices = self._resolve_picks(picks)

        if self.data.ndim == 2:
            data = self.data[indices, :]
        else:
            data = self.data[:, indices, :]

        return aplicar_hilbert(data)

    def hilbert_transform(self, picks=None):
        """
            Calcula la transformada de Hilbert para uno o mas canales.

        El calculo real se delega al modulo preprocesamiento/filtros.py.
        """

        from .preprocesamiento.filtros import aplicar_hilbert

        indices = self._resolve_picks(picks)

        if self.data.ndim == 2:
            data = self.data[indices, :]
        else:
            data = self.data[:, indices, :]

        return aplicar_hilbert(data)

    def _resolve_picks(
        self,
        picks: Optional[Sequence[int | str] | int | str],
    ) -> list[int]:
        """
        Convierte nombres o indices de canales a indices.
        """

        if picks is None:
            return list(range(self.n_channels))

        if isinstance(picks, (int, str)):
            picks = [picks]

        indices = []

        for pick in picks:
            if isinstance(pick, int):
                if pick < 0 or pick >= self.n_channels:
                    raise ValueError("Indice de canal fuera de rango.")

                indices.append(pick)

            elif isinstance(pick, str):
                if pick not in self.ch_names:
                    raise ValueError(f"Canal no encontrado: {pick}")

                indices.append(self.ch_names.index(pick))

            else:
                raise TypeError("picks debe contener indices o nombres de canales.")

        return indices

    def _crop_annotations(self, tmin: float, tmax: float) -> Anotaciones:
        """
        Recorta anotaciones segun el intervalo temporal seleccionado.
        """

        if self.anotaciones is None or len(self.anotaciones) == 0:
            return Anotaciones()

        df = self.anotaciones.get_annotations()
        mask = (df["onset"] >= tmin) & (df["onset"] <= tmax)
        df = df[mask].copy()

        if len(df) == 0:
            return Anotaciones()

        df["onset"] = df["onset"] - tmin

        ch_names = None

        if "ch_names" in df.columns:
            ch_names = df["ch_names"].tolist()

        return Anotaciones(
            onset=df["onset"].tolist(),
            duration=df["duration"].tolist(),
            description=df["description"].tolist(),
            ch_names=ch_names,
        )

    def _new_like(
        self,
        data: np.ndarray,
        ch_names: Optional[Sequence[str]] = None,
        ch_types: Optional[Sequence[str]] = None,
        times: Optional[np.ndarray] = None,
        montage: Optional[dict] = None,
        reference: Optional[str] = None,
        units: Optional[str] = None,
        anotaciones: Optional[Anotaciones] = None,
        eventos: Optional[Eventos] = None,
        event_id: Optional[dict[str, int]] = None,
        subject_info: Optional[dict] = None,
        meas_date: Optional[str] = None,
        filters: Optional[list[dict]] = None,
        is_epoched: Optional[bool] = None,
        is_filtered: Optional[bool] = None,
        first_samp: Optional[int] = None,
    ) -> "EEGSignal":
        """
        Crea una nueva instancia conservando la metadata del objeto actual.
        """

        return EEGSignal(
            data=data,
            sfreq=self.sfreq,
            ch_names=ch_names if ch_names is not None else self.ch_names,
            ch_types=ch_types if ch_types is not None else self.ch_types,
            times=times if times is not None else self.times.copy(),
            montage=montage if montage is not None else copy.deepcopy(self.montage),
            reference=reference if reference is not None else self.reference,
            units=units if units is not None else self.units,
            anotaciones=anotaciones if anotaciones is not None else copy.deepcopy(self.anotaciones),
            eventos=eventos if eventos is not None else copy.deepcopy(self.eventos),
            event_id=event_id if event_id is not None else copy.deepcopy(self.event_id),
            subject_info=subject_info if subject_info is not None else copy.deepcopy(self.subject_info),
            meas_date=meas_date if meas_date is not None else self.meas_date,
            filters=filters if filters is not None else copy.deepcopy(self.filters),
            is_epoched=is_epoched if is_epoched is not None else self.is_epoched,
            is_filtered=is_filtered if is_filtered is not None else self.is_filtered,
            first_samp=first_samp if first_samp is not None else self.first_samp,
        )

    def __len__(self) -> int:
        """
        Devuelve la cantidad de muestras.
        """

        return self.n_samples

    def __repr__(self) -> str:
        if self.data.ndim == 2:
            return (
                f"EEGSignal(n_channels={self.n_channels}, "
                f"n_samples={self.n_samples}, "
                f"sfreq={self.sfreq}, "
                f"duration={self.duration:.3f} s)"
            )

        return (
            f"EEGSignal(n_trials={self.n_trials}, "
            f"n_channels={self.n_channels}, "
            f"n_samples={self.n_samples}, "
            f"sfreq={self.sfreq}, "
            f"duration={self.duration:.3f} s)"
        )