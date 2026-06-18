"""
Tests de la clase RawSignal.

Cubren la checklist de la seccion 5.3 del documento del TPI: creacion,
acceso a datos, picks, recorte de muestras, tiempos, reject, drop_channels,
crop, resumen, describe, get_channel, pick_channels, pick_types,
set_anotaciones, __getitem__, duracion, __str__ y los casos de error
esperados (canal inexistente, indices invalidos, tmax > duracion,
dimensiones incompatibles con la metadata).

NOTA: drop_channels, crop, pick_channels y pick_types modifican la
instancia in-place, por eso cada test crea un RawSignal nuevo.
"""

import numpy as np
import pytest

from tpi_analisisdesenales.info import Info
from tpi_analisisdesenales.signals import RawSignal
from tpi_analisisdesenales.eventos import Anotaciones


SFREQ = 100.0
N_SAMPLES = 100


def crear_raw(ch_types=None):
    """
    Crea un RawSignal de 3 canales con datos deterministicos.

    data[i] = [i*100, i*100+1, ..., i*100+99]
    Asi cada canal tiene un rango pico a pico de 99.
    """
    if ch_types is None:
        ch_types = ["eeg", "eeg", "eeg"]

    info = Info(
        ch_names=["C3", "Cz", "C4"],
        sfreq=SFREQ,
        ch_types=ch_types,
    )
    data = np.arange(3 * N_SAMPLES, dtype=float).reshape(3, N_SAMPLES)
    return RawSignal(data=data, info=info)


# --------------------------------------------------------------------------
# Creacion
# --------------------------------------------------------------------------


def test_creacion_desde_ndarray_e_info():
    raw = crear_raw()

    assert raw.data.shape == (3, N_SAMPLES)
    assert raw.sfreq == SFREQ
    assert raw.info.n_samples == N_SAMPLES  # se completa al construir
    assert raw.anotaciones is not None
    assert raw.eventos is not None


# --------------------------------------------------------------------------
# get_data: completo, picks, start/stop, times, reject
# --------------------------------------------------------------------------


def test_get_data_completo():
    raw = crear_raw()

    out = raw.get_data()

    assert out.shape == (3, N_SAMPLES)
    assert np.allclose(out, raw.data)


def test_get_data_con_picks():
    raw = crear_raw()

    assert raw.get_data(picks=["C3", "C4"]).shape == (2, N_SAMPLES)
    assert raw.get_data(picks="Cz").shape == (1, N_SAMPLES)


def test_get_data_recorte_de_muestras():
    raw = crear_raw()

    out = raw.get_data(start=0, stop=50)

    assert out.shape == (3, 50)


def test_get_data_devuelve_tiempos():
    raw = crear_raw()

    out, t = raw.get_data(times=True)

    assert out.shape == (3, N_SAMPLES)
    assert len(t) == N_SAMPLES
    assert t[0] == pytest.approx(0.0)
    assert t[1] == pytest.approx(1 / SFREQ)


def test_get_data_reject_descarta_por_amplitud():
    info = Info(ch_names=["bajo", "alto"], sfreq=SFREQ, ch_types=["eeg", "eeg"])
    data = np.zeros((2, N_SAMPLES))
    data[0] = np.linspace(0, 10, N_SAMPLES)     # ptp = 10
    data[1] = np.linspace(0, 1000, N_SAMPLES)   # ptp = 1000
    raw = RawSignal(data=data, info=info)

    out = raw.get_data(reject=100)

    # solo debe quedar el canal de bajo pico a pico
    assert out.shape == (1, N_SAMPLES)


# --------------------------------------------------------------------------
# drop_channels y crop (in-place)
# --------------------------------------------------------------------------


def test_drop_channels():
    raw = crear_raw()

    retorno = raw.drop_channels(["Cz"])

    assert retorno is raw  # modifica in-place y devuelve self
    assert raw.info.ch_names == ["C3", "C4"]
    assert raw.get_data().shape == (2, N_SAMPLES)


def test_crop_recorte_temporal():
    raw = crear_raw()

    raw.crop(tmin=0.0, tmax=0.5)

    assert raw.get_data().shape == (3, 50)


# --------------------------------------------------------------------------
# resumen y describe
# --------------------------------------------------------------------------


def test_resumen_tiene_las_claves_esperadas():
    raw = crear_raw()

    resumen = raw.resumen()

    for clave in [
        "n_channels",
        "n_samples",
        "Frecuencia de muestreo",
        "Duracion",
        "Referencia",
        "Filtros aplicados",
    ]:
        assert clave in resumen

    assert resumen["n_channels"] == 3
    assert resumen["n_samples"] == N_SAMPLES


def test_describe_estadisticas_por_canal():
    raw = crear_raw()

    df = raw.describe()

    assert df.shape[0] == 3
    assert list(df["name"]) == ["C3", "Cz", "C4"]
    for col in ["min", "Q1", "mediana", "Q3", "max", "rms", "var"]:
        assert col in df.columns


# --------------------------------------------------------------------------
# get_channel, pick_channels, pick_types
# --------------------------------------------------------------------------


def test_get_channel_devuelve_array_1d():
    raw = crear_raw()

    canal = raw.get_channel("Cz")

    assert canal.ndim == 1
    assert len(canal) == N_SAMPLES
    assert np.allclose(canal, raw.data[1])


def test_pick_channels():
    raw = crear_raw()

    raw.pick_channels(["C4"])

    assert raw.info.ch_names == ["C4"]
    assert raw.get_data().shape == (1, N_SAMPLES)


def test_pick_types():
    raw = crear_raw(ch_types=["eeg", "eeg", "ecg"])

    raw.pick_types(eeg=True)

    assert raw.info.ch_names == ["C3", "Cz"]
    assert all(t == "eeg" for t in raw.info.ch_types)


# --------------------------------------------------------------------------
# set_anotaciones
# --------------------------------------------------------------------------


def test_set_anotaciones():
    raw = crear_raw()

    anot = Anotaciones()
    anot.add(onset=1.0, duration=0.5, description="Artefacto")
    raw.set_anotaciones(anot)

    assert len(raw.anotaciones) == 1
    assert raw.anotaciones[0]["description"] == "Artefacto"


# --------------------------------------------------------------------------
# __getitem__
# --------------------------------------------------------------------------


def test_getitem_un_canal():
    raw = crear_raw()

    c3, t = raw["C3"]

    assert c3.shape == (1, N_SAMPLES)
    assert len(t) == N_SAMPLES


def test_getitem_canal_y_slice():
    raw = crear_raw()

    cz, t = raw["Cz", 0:50]

    assert cz.shape == (1, 50)
    assert len(t) == 50


def test_getitem_lista_de_canales():
    raw = crear_raw()

    datos, t = raw[["C3", "C4"], :]

    assert datos.shape == (2, N_SAMPLES)


# --------------------------------------------------------------------------
# duracion y __str__
# --------------------------------------------------------------------------


def test_duration():
    raw = crear_raw()

    assert raw.duration == pytest.approx(N_SAMPLES / SFREQ)


def test_str():
    raw = crear_raw()

    texto = str(raw)

    assert "RawSignal" in texto
    assert "n_channels=3" in texto


# --------------------------------------------------------------------------
# Casos de error
# --------------------------------------------------------------------------


def test_error_canal_inexistente():
    raw = crear_raw()

    with pytest.raises(ValueError):
        raw.get_data(picks="NoExiste")


def test_error_indices_invalidos():
    raw = crear_raw()

    # stop mayor que la cantidad de muestras
    with pytest.raises(ValueError):
        raw.get_data(start=0, stop=999)

    # indice de canal fuera de rango
    with pytest.raises(IndexError):
        raw.get_data(picks=99)


def test_error_tmax_mayor_que_duracion():
    raw = crear_raw()

    with pytest.raises(ValueError):
        raw.crop(tmin=0.0, tmax=999.0)


def test_error_dimensiones_incompatibles_con_metadata():
    # data con 2 canales pero info declara 3
    info = Info(ch_names=["C3", "Cz", "C4"], sfreq=SFREQ, ch_types=["eeg"] * 3)
    data = np.zeros((2, N_SAMPLES))

    with pytest.raises(ValueError):
        RawSignal(data=data, info=info)


def test_error_data_no_2d():
    info = Info(ch_names=["C3"], sfreq=SFREQ, ch_types=["eeg"])
    data = np.zeros(N_SAMPLES)  # 1D

    with pytest.raises(ValueError):
        RawSignal(data=data, info=info)


def test_error_set_anotaciones_tipo_invalido():
    raw = crear_raw()

    with pytest.raises(TypeError):
        raw.set_anotaciones("no soy Anotaciones")
