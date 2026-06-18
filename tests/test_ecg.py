"""
Tests de la clase ECGSignal.

Se verifican: construccion, deteccion de picos R, frecuencia cardiaca,
filtrado, estadisticas heredadas y casos de error.
"""

import numpy as np
import pandas as pd
import pytest

from tpi_analisisdesenales.info import Info
from tpi_analisisdesenales.signals import ECGSignal


SFREQ = 250.0


def _make_ecg_sintetico(bpm=120, segundos=8):
    """
    Genera un ECG sintetico con picos R periodicos bien marcados.

    Con bpm=120 hay un latido cada 0.5 s.
    """
    n = int(SFREQ * segundos)
    senal = np.zeros(n)

    paso = int(SFREQ * 60.0 / bpm)  # muestras entre latidos
    posiciones = np.arange(paso, n, paso)
    senal[posiciones] = 10.0  # picos R

    # un poco de ruido suave
    senal = senal + 0.01 * np.random.default_rng(0).standard_normal(n)

    data = senal.reshape(1, -1)
    info = Info(ch_names=["DII"], sfreq=SFREQ, ch_types=["ecg"])

    return ECGSignal(data=data, info=info), len(posiciones)


# --------------------------------------------------------------------------
# Construccion
# --------------------------------------------------------------------------


def test_construccion():
    ecg, _ = _make_ecg_sintetico()

    assert ecg.tipo_senal == "ECG"
    assert ecg.info.ch_names == ["DII"]
    assert ecg.sfreq == SFREQ


# --------------------------------------------------------------------------
# Deteccion de picos R
# --------------------------------------------------------------------------


def test_detectar_picos_r():
    ecg, n_picos_esperados = _make_ecg_sintetico()

    picos = ecg.detectar_picos_r()

    assert isinstance(picos, np.ndarray)
    # se deben detectar (casi) todos los picos generados
    assert len(picos) == n_picos_esperados


def test_frecuencia_cardiaca_aproximada():
    ecg, _ = _make_ecg_sintetico(bpm=120)

    hr = ecg.calcular_frecuencia_cardiaca()

    assert hr == pytest.approx(120, abs=2)


def test_detectar_picos_r_canal_fuera_de_rango():
    ecg, _ = _make_ecg_sintetico()

    with pytest.raises(ValueError, match="canal"):
        ecg.detectar_picos_r(canal=5)


# --------------------------------------------------------------------------
# Filtrado
# --------------------------------------------------------------------------


def test_filtrar_devuelve_ecg():
    ecg, _ = _make_ecg_sintetico()

    filtrada = ecg.filtrar(l_freq=0.5, h_freq=40.0, notch_freq=50.0)

    assert isinstance(filtrada, ECGSignal)
    assert filtrada.data.shape == ecg.data.shape


# --------------------------------------------------------------------------
# Estadisticas heredadas de RawSignal
# --------------------------------------------------------------------------


def test_describe_heredado():
    ecg, _ = _make_ecg_sintetico()

    df = ecg.describe()

    assert isinstance(df, pd.DataFrame)
    assert list(df["name"]) == ["DII"]
    assert "rms" in df.columns


def test_describe_ecg_resumen():
    ecg, _ = _make_ecg_sintetico()

    resumen = ecg.describe_ecg()

    assert resumen["tipo_senal"] == "ECG"
    assert resumen["n_picos_r"] > 0
    assert resumen["heart_rate_bpm"] == pytest.approx(120, abs=2)
