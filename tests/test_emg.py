"""
Tests de la clase EMGSignal.

Se verifican: construccion y validacion estructural, calculo de RMS, MAV
y varianza por canal, envolvente/rectificacion, extraccion de caracteristicas,
estadisticas y casos de error.
"""

import numpy as np
import pandas as pd
import pytest

from tpi_analisisdesenales.info import Info
from tpi_analisisdesenales.signals import EMGSignal


SFREQ = 1000.0
N_SAMPLES = 2000


def _make_emg():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((2, N_SAMPLES))
    info = Info(
        ch_names=["biceps", "triceps"],
        sfreq=SFREQ,
        ch_types=["emg", "emg"],
    )
    return EMGSignal(data=data, info=info, muscle_group="brazo")


# --------------------------------------------------------------------------
# Construccion y validacion
# --------------------------------------------------------------------------


def test_construccion():
    emg = _make_emg()

    assert emg.tipo_senal == "EMG"
    assert emg.muscle_group == "brazo"
    assert emg.info.ch_names == ["biceps", "triceps"]


def test_validate_structure_ok():
    emg = _make_emg()

    assert emg.validate_structure() is True


# --------------------------------------------------------------------------
# Caracteristicas por canal
# --------------------------------------------------------------------------


def test_rms_por_canal():
    emg = _make_emg()

    rms = emg.rms_per_channel()

    assert rms.shape == (2,)
    assert np.all(rms > 0)


def test_mav_y_varianza_por_canal():
    emg = _make_emg()

    mav = emg.mav_per_channel()
    var = emg.variance_per_channel()

    assert mav.shape == (2,)
    assert var.shape == (2,)
    assert np.all(mav >= 0)
    assert np.all(var >= 0)


def test_envolvente_y_rectificacion_no_negativas():
    emg = _make_emg()

    env = emg.envelope()
    rect = emg.rectificar()

    assert np.all(env >= 0)
    assert np.all(rect >= 0)
    assert env.shape == emg.data.shape


def test_feature_extraction_devuelve_diccionario():
    emg = _make_emg()

    features = emg.feature_extraction()

    assert set(features.keys()) == {"rms", "mav", "var"}
    assert features["rms"].shape == (2,)


# --------------------------------------------------------------------------
# Estadisticas
# --------------------------------------------------------------------------


def test_describe_agrega_columnas_emg():
    emg = _make_emg()

    df = emg.describe()

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2
    for col in ["emg_rms", "emg_mav", "emg_var"]:
        assert col in df.columns


def test_filtrar_devuelve_emg():
    emg = _make_emg()

    filtrada = emg.filtrar(l_freq=20.0, h_freq=150.0, notch_freq=50.0)

    assert isinstance(filtrada, EMGSignal)
    assert filtrada.data.shape == emg.data.shape


# --------------------------------------------------------------------------
# Errores
# --------------------------------------------------------------------------


def test_error_ch_names_no_coincide():
    data = np.random.randn(2, N_SAMPLES)
    info = Info(ch_names=["solo_uno"], sfreq=SFREQ, ch_types=["emg"])

    # data tiene 2 canales pero info solo 1 -> RawSignal debe quejarse
    with pytest.raises(ValueError):
        EMGSignal(data=data, info=info)
