"""
Tests de graficado segun el tipo de senal (ECG, EEG y EMG).

El camino de graficado (plot_raw -> PlotEngine) es comun a las tres clases,
por lo que aqui solo se verifica lo que SI depende del tipo de senal:
que cada clase grafique sus propios canales (cantidad y nombres correctos).

El comportamiento general del motor de graficado (figura valida, superposicion,
rango temporal, media +- desvio, anotaciones) se prueba en test_plot_engine.py.

Todas las pruebas usan show=False para no abrir el navegador.
"""

import numpy as np
import pytest

from tpi_analisisdesenales.info import Info
from tpi_analisisdesenales.signals import EEGSignal, ECGSignal, EMGSignal
from tpi_analisisdesenales.visualizacion import plot_raw


SFREQ = 250.0
N_SAMPLES = 1000


def _make_eeg():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((3, N_SAMPLES))
    return EEGSignal(
        data=data,
        sfreq=SFREQ,
        ch_names=["C3", "Cz", "C4"],
        ch_types=["eeg", "eeg", "eeg"],
    )


def _make_ecg():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((2, N_SAMPLES))
    info = Info(ch_names=["DI", "DII"], sfreq=SFREQ, ch_types=["ecg", "ecg"])
    return ECGSignal(data=data, info=info)


def _make_emg():
    rng = np.random.default_rng(2)
    data = rng.standard_normal((2, N_SAMPLES))
    info = Info(ch_names=["biceps", "triceps"], sfreq=SFREQ, ch_types=["emg", "emg"])
    return EMGSignal(data=data, info=info)


# (nombre_tipo, builder, nombres_de_canal_esperados)
SIGNAL_CASES = [
    ("EEG", _make_eeg, ["C3", "Cz", "C4"]),
    ("ECG", _make_ecg, ["DI", "DII"]),
    ("EMG", _make_emg, ["biceps", "triceps"]),
]


@pytest.mark.parametrize("tipo, builder, ch_names", SIGNAL_CASES)
def test_un_trazo_por_canal(tipo, builder, ch_names):
    """
    Cada tipo de senal debe graficar un trazo por cada uno de sus canales.
    """
    signal = builder()

    fig = plot_raw(signal, show=False)

    assert len(fig.data) == len(ch_names), (
        f"{tipo}: se esperaban {len(ch_names)} trazos y hay {len(fig.data)}"
    )


@pytest.mark.parametrize("tipo, builder, ch_names", SIGNAL_CASES)
def test_nombres_de_trazos_coinciden_con_canales(tipo, builder, ch_names):
    """
    Los trazos deben llevar los nombres de canal de cada tipo de senal.
    Esto confirma que se grafica la senal correcta segun su metadata.
    """
    signal = builder()

    fig = plot_raw(signal, show=False)

    nombres_trazos = [trace.name for trace in fig.data]

    assert nombres_trazos == ch_names, (
        f"{tipo}: nombres de trazos {nombres_trazos} != canales {ch_names}"
    )


def test_cada_tipo_grafica_su_cantidad_de_canales():
    """
    Vista cruzada: cada tipo de senal genera la cantidad de trazos correcta.
    """
    esperado = {"EEG": 3, "ECG": 2, "EMG": 2}

    figuras = {
        "EEG": plot_raw(_make_eeg(), show=False),
        "ECG": plot_raw(_make_ecg(), show=False),
        "EMG": plot_raw(_make_emg(), show=False),
    }

    for tipo, fig in figuras.items():
        assert len(fig.data) == esperado[tipo], (
            f"{tipo}: se esperaban {esperado[tipo]} trazos y hay {len(fig.data)}"
        )
