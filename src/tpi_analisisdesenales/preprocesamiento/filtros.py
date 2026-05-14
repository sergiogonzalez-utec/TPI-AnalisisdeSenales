"""
Modulo de filtros y transformaciones para senales biologicas.

Incluye:
- Filtro pasabanda
- Filtro pasa bajos
- Filtro pasa altos
- Filtro notch
- Transformada de Hilbert

Las funciones trabajan con datos 2D o 3D:
- 2D: n_canales x n_muestras
- 3D: n_trials x n_canales x n_muestras
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, hilbert


def _validar_data(data: np.ndarray) -> np.ndarray:
    """
    Valida que data sea un arreglo 2D o 3D.
    """

    data = np.asarray(data)

    if data.ndim not in (2, 3):
        raise ValueError(
            "data debe tener forma n_canales x n_muestras "
            "o n_trials x n_canales x n_muestras."
        )

    return data


def _validar_sfreq(sfreq: float) -> float:
    """
    Valida la frecuencia de muestreo.
    """

    sfreq = float(sfreq)

    if sfreq <= 0:
        raise ValueError("sfreq debe ser mayor que cero.")

    return sfreq


def filtro_pasabanda(
    data: np.ndarray,
    sfreq: float,
    l_freq: float,
    h_freq: float,
    order: int = 4,
) -> np.ndarray:
    """
    Aplica un filtro pasabanda Butterworth.

    Parameters
    ----------
    data : np.ndarray
        Datos con forma n_canales x n_muestras
        o n_trials x n_canales x n_muestras.

    sfreq : float
        Frecuencia de muestreo en Hz.

    l_freq : float
        Frecuencia de corte inferior en Hz.

    h_freq : float
        Frecuencia de corte superior en Hz.

    order : int
        Orden del filtro.

    Returns
    -------
    data_filtrada : np.ndarray
        Senal filtrada.
    """

    data = _validar_data(data)
    sfreq = _validar_sfreq(sfreq)

    nyquist = sfreq / 2.0

    if l_freq <= 0:
        raise ValueError("l_freq debe ser mayor que cero.")

    if h_freq >= nyquist:
        raise ValueError("h_freq debe ser menor que la frecuencia de Nyquist.")

    if l_freq >= h_freq:
        raise ValueError("l_freq debe ser menor que h_freq.")

    if order <= 0:
        raise ValueError("order debe ser mayor que cero.")

    low = l_freq / nyquist
    high = h_freq / nyquist

    b, a = butter(order, [low, high], btype="bandpass")

    return filtfilt(b, a, data, axis=-1)


def filtro_pasabajos(
    data: np.ndarray,
    sfreq: float,
    h_freq: float,
    order: int = 4,
) -> np.ndarray:
    """
    Aplica un filtro pasa bajos Butterworth.
    """

    data = _validar_data(data)
    sfreq = _validar_sfreq(sfreq)

    nyquist = sfreq / 2.0

    if h_freq <= 0:
        raise ValueError("h_freq debe ser mayor que cero.")

    if h_freq >= nyquist:
        raise ValueError("h_freq debe ser menor que la frecuencia de Nyquist.")

    if order <= 0:
        raise ValueError("order debe ser mayor que cero.")

    cutoff = h_freq / nyquist

    b, a = butter(order, cutoff, btype="lowpass")

    return filtfilt(b, a, data, axis=-1)


def filtro_pasaaltos(
    data: np.ndarray,
    sfreq: float,
    l_freq: float,
    order: int = 4,
) -> np.ndarray:
    """
    Aplica un filtro pasa altos Butterworth.
    """

    data = _validar_data(data)
    sfreq = _validar_sfreq(sfreq)

    nyquist = sfreq / 2.0

    if l_freq <= 0:
        raise ValueError("l_freq debe ser mayor que cero.")

    if l_freq >= nyquist:
        raise ValueError("l_freq debe ser menor que la frecuencia de Nyquist.")

    if order <= 0:
        raise ValueError("order debe ser mayor que cero.")

    cutoff = l_freq / nyquist

    b, a = butter(order, cutoff, btype="highpass")

    return filtfilt(b, a, data, axis=-1)


def filtro_notch(
    data: np.ndarray,
    sfreq: float,
    freq: float = 50.0,
    q: float = 30.0,
) -> np.ndarray:
    """
    Aplica un filtro notch para atenuar ruido de red.

    Parameters
    ----------
    data : np.ndarray
        Datos con forma n_canales x n_muestras
        o n_trials x n_canales x n_muestras.

    sfreq : float
        Frecuencia de muestreo en Hz.

    freq : float
        Frecuencia a eliminar. Usualmente 50 Hz o 60 Hz.

    q : float
        Factor de calidad del filtro.

    Returns
    -------
    data_filtrada : np.ndarray
        Senal filtrada.
    """

    data = _validar_data(data)
    sfreq = _validar_sfreq(sfreq)

    nyquist = sfreq / 2.0

    if freq <= 0:
        raise ValueError("freq debe ser mayor que cero.")

    if freq >= nyquist:
        raise ValueError("freq debe ser menor que la frecuencia de Nyquist.")

    if q <= 0:
        raise ValueError("q debe ser mayor que cero.")

    b, a = iirnotch(w0=freq, Q=q, fs=sfreq)

    return filtfilt(b, a, data, axis=-1)


def aplicar_hilbert(data: np.ndarray):
    """
    Aplica la transformada de Hilbert.

    Parameters
    ----------
    data : np.ndarray
        Datos con forma n_canales x n_muestras
        o n_trials x n_canales x n_muestras.

    Returns
    -------
    analytic_signal : np.ndarray
        Senal analitica compleja.

    envelope : np.ndarray
        Envolvente de amplitud.

    phase : np.ndarray
        Fase instantanea.
    """

    data = _validar_data(data)

    analytic_signal = hilbert(data, axis=-1)
    envelope = np.abs(analytic_signal)
    phase = np.angle(analytic_signal)

    return analytic_signal, envelope, phase


def frecuencia_instantanea(phase: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Calcula la frecuencia instantanea a partir de la fase.

    Parameters
    ----------
    phase : np.ndarray
        Fase instantanea obtenida con Hilbert.

    sfreq : float
        Frecuencia de muestreo en Hz.

    Returns
    -------
    inst_freq : np.ndarray
        Frecuencia instantanea en Hz.
    """

    sfreq = _validar_sfreq(sfreq)

    unwrapped_phase = np.unwrap(phase, axis=-1)
    inst_freq = np.diff(unwrapped_phase, axis=-1) * sfreq / (2.0 * np.pi)

    return inst_freq