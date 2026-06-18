"""
Tests de la clase EEGSignal.

Se verifican casos validos (construccion, acceso a datos, recorte, seleccion
de canales, referencia, filtrado, estadisticas) y casos de error.
"""

import numpy as np
import pandas as pd
import pytest

from tpi_analisisdesenales.signals import EEGSignal


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


# --------------------------------------------------------------------------
# Construccion y propiedades
# --------------------------------------------------------------------------


def test_construccion_y_propiedades():
    eeg = _make_eeg()

    assert eeg.tipo_senal == "EEG"
    assert eeg.n_channels == 3
    assert eeg.n_samples == N_SAMPLES
    assert eeg.duration == pytest.approx(N_SAMPLES / SFREQ)
    assert eeg.info.ch_names == ["C3", "Cz", "C4"]


def test_times_se_genera_automaticamente():
    eeg = _make_eeg()

    assert len(eeg.times) == N_SAMPLES
    assert eeg.times[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Acceso a datos
# --------------------------------------------------------------------------


def test_get_data_todos_los_canales():
    eeg = _make_eeg()

    out = eeg.get_data()

    assert out.shape == (3, N_SAMPLES)


def test_get_data_con_picks_y_rango():
    eeg = _make_eeg()

    out = eeg.get_data(picks=["C3", "C4"], start=0, stop=500)

    assert out.shape == (2, 500)


def test_get_channels_devuelve_nueva_instancia():
    eeg = _make_eeg()

    sub = eeg.get_channels(["C3"])

    assert isinstance(sub, EEGSignal)
    assert sub is not eeg
    assert sub.info.ch_names == ["C3"]
    # la original no se modifica
    assert eeg.n_channels == 3


# --------------------------------------------------------------------------
# Recorte temporal
# --------------------------------------------------------------------------


def test_crop_devuelve_nueva_instancia():
    eeg = _make_eeg()

    recorte = eeg.crop(tmin=0.0, tmax=1.0)

    assert isinstance(recorte, EEGSignal)
    assert recorte is not eeg
    assert recorte.n_samples == 250
    assert eeg.n_samples == N_SAMPLES  # original intacta


def test_crop_intervalo_invalido():
    eeg = _make_eeg()

    with pytest.raises(ValueError):
        eeg.crop(tmin=0.0, tmax=999.0)


# --------------------------------------------------------------------------
# Referencia y descarte de canales
# --------------------------------------------------------------------------


def test_set_reference_resta_el_canal():
    eeg = _make_eeg()

    referenciada = eeg.set_reference("Cz")

    cz_idx = referenciada.info.ch_names.index("Cz")
    # el canal de referencia debe quedar en cero
    assert np.allclose(referenciada.data[cz_idx], 0.0)


def test_drop_channels_devuelve_nueva_instancia():
    eeg = _make_eeg()

    nueva = eeg.drop_channels(["Cz"])

    assert nueva.info.ch_names == ["C3", "C4"]
    assert nueva.n_channels == 2
    assert eeg.n_channels == 3


# --------------------------------------------------------------------------
# Filtrado
# --------------------------------------------------------------------------


def test_filter_no_modifica_la_original():
    eeg = _make_eeg()
    datos_originales = eeg.data.copy()

    filtrada = eeg.filter(l_freq=1.0, h_freq=40.0)

    assert isinstance(filtrada, EEGSignal)
    assert filtrada.is_filtered is True
    assert np.allclose(eeg.data, datos_originales)  # original sin cambios


def test_filtrar_agrega_notch_y_pasabanda():
    eeg = _make_eeg()

    filtrada = eeg.filtrar(l_freq=1.0, h_freq=40.0, notch_freq=50.0)

    tipos = [f["type"] for f in filtrada.filters]
    assert "notch" in tipos
    assert "bandpass" in tipos


# --------------------------------------------------------------------------
# Estadisticas
# --------------------------------------------------------------------------


def test_describe_tiene_una_fila_por_canal():
    eeg = _make_eeg()

    df = eeg.describe()

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 3
    assert list(df["name"]) == ["C3", "Cz", "C4"]
    for col in ["min", "max", "rms", "std", "median"]:
        assert col in df.columns


# --------------------------------------------------------------------------
# Errores de construccion
# --------------------------------------------------------------------------


def test_error_ch_names_no_coincide():
    data = np.random.randn(3, N_SAMPLES)

    with pytest.raises(ValueError, match="ch_names"):
        EEGSignal(data=data, sfreq=SFREQ, ch_names=["C3", "C4"])


def test_error_sfreq_invalida():
    data = np.random.randn(2, N_SAMPLES)

    with pytest.raises(ValueError, match="sfreq"):
        EEGSignal(data=data, sfreq=0, ch_names=["C3", "C4"])


def test_error_canal_inexistente_en_get_data():
    eeg = _make_eeg()

    with pytest.raises(ValueError):
        eeg.get_data(picks=["NoExiste"])
