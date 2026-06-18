"""
Tests de graficado segun el tipo de senal (ECG, EEG y EMG).

La idea de estos tests es verificar que, sin importar el tipo de senal,
el grafico se construya correctamente:

- se genera una figura de Plotly valida,
- hay un trazo (linea) por cada canal graficado,
- los nombres de los trazos coinciden con los nombres de canal,
- la seleccion de canales (picks) reduce los trazos,
- la superposicion sigue dibujando un trazo por canal,
- la media +- desvio genera una figura valida,
- las anotaciones se dibujan como formas (shapes) sobre la figura.

Todas las pruebas usan show=False para no abrir el navegador.

Cada tipo de senal se grafica a traves de plot_raw, que es exactamente
la funcion en la que delegan RawSignal.plot() y EEGSignal.plot().
"""

import numpy as np
import plotly.graph_objects as go
import pytest

from tpi_analisisdesenales.info import Info
from tpi_analisisdesenales.eventos import Anotaciones
from tpi_analisisdesenales.signals import RawSignal, EEGSignal, ECGSignal, EMGSignal
from tpi_analisisdesenales.visualizacion import PlotEngine, plot_raw


# ---------------------------------------------------------------------------
# Constructores de apoyo: una senal por cada tipo (EEG, ECG, EMG).
# Se usa una semilla fija para que los datos sean reproducibles.
# ---------------------------------------------------------------------------

SFREQ = 250.0
N_SAMPLES = 1000  # 4 segundos a 250 Hz


def _make_eeg():
    """Crea una EEGSignal de 3 canales."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((3, N_SAMPLES))
    ch_names = ["C3", "Cz", "C4"]

    return EEGSignal(
        data=data,
        sfreq=SFREQ,
        ch_names=ch_names,
        ch_types=["eeg", "eeg", "eeg"],
    )


def _make_ecg():
    """Crea una ECGSignal de 2 canales (derivaciones)."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((2, N_SAMPLES))
    ch_names = ["DI", "DII"]

    info = Info(
        ch_names=ch_names,
        sfreq=SFREQ,
        ch_types=["ecg", "ecg"],
    )

    return ECGSignal(data=data, info=info)


def _make_emg():
    """Crea una EMGSignal de 2 canales (musculos)."""
    rng = np.random.default_rng(2)
    data = rng.standard_normal((2, N_SAMPLES))
    ch_names = ["biceps", "triceps"]

    info = Info(
        ch_names=ch_names,
        sfreq=SFREQ,
        ch_types=["emg", "emg"],
    )

    return EMGSignal(data=data, info=info)


# Lista parametrizada: (nombre_tipo, builder, nombres_de_canal_esperados)
SIGNAL_CASES = [
    ("EEG", _make_eeg, ["C3", "Cz", "C4"]),
    ("ECG", _make_ecg, ["DI", "DII"]),
    ("EMG", _make_emg, ["biceps", "triceps"]),
]


# ---------------------------------------------------------------------------
# Tests basicos de graficado por tipo de senal.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tipo, builder, ch_names", SIGNAL_CASES)
def test_plot_devuelve_figura_plotly(tipo, builder, ch_names):
    """
    Cualquier tipo de senal debe producir una figura de Plotly valida.
    """
    signal = builder()

    fig = plot_raw(signal, show=False)

    assert isinstance(fig, go.Figure), f"{tipo}: no se devolvio una figura Plotly"
    assert hasattr(fig, "data")
    assert hasattr(fig, "layout")


@pytest.mark.parametrize("tipo, builder, ch_names", SIGNAL_CASES)
def test_un_trazo_por_canal(tipo, builder, ch_names):
    """
    Sin picks, debe haber exactamente un trazo por cada canal de la senal.
    """
    signal = builder()

    fig = plot_raw(signal, show=False)

    assert len(fig.data) == len(ch_names), (
        f"{tipo}: se esperaban {len(ch_names)} trazos y hay {len(fig.data)}"
    )


@pytest.mark.parametrize("tipo, builder, ch_names", SIGNAL_CASES)
def test_nombres_de_trazos_coinciden_con_canales(tipo, builder, ch_names):
    """
    Cada trazo debe llevar el nombre del canal correspondiente.
    Esto confirma que se grafica la senal correcta segun su metadata.
    """
    signal = builder()

    fig = plot_raw(signal, show=False)

    nombres_trazos = [trace.name for trace in fig.data]

    assert nombres_trazos == ch_names, (
        f"{tipo}: nombres de trazos {nombres_trazos} != canales {ch_names}"
    )


@pytest.mark.parametrize("tipo, builder, ch_names", SIGNAL_CASES)
def test_picks_selecciona_un_solo_canal(tipo, builder, ch_names):
    """
    Al elegir un canal por nombre, debe graficarse un unico trazo.
    """
    signal = builder()
    canal = ch_names[0]

    fig = plot_raw(signal, picks=[canal], show=False)

    assert len(fig.data) == 1
    assert fig.data[0].name == canal


@pytest.mark.parametrize("tipo, builder, ch_names", SIGNAL_CASES)
def test_superpose_mantiene_un_trazo_por_canal(tipo, builder, ch_names):
    """
    En modo superpuesto se siguen dibujando todos los canales,
    cada uno como un trazo independiente (para que tengan colores distintos).
    """
    signal = builder()

    fig = plot_raw(signal, superpose=True, show=False)

    assert len(fig.data) == len(ch_names)


@pytest.mark.parametrize("tipo, builder, ch_names", SIGNAL_CASES)
def test_rango_temporal_recorta_el_eje_x(tipo, builder, ch_names):
    """
    Si se pide una ventana de tiempo, los trazos no deben exceder ese rango.
    """
    signal = builder()

    fig = plot_raw(signal, start=0.0, stop=2.0, show=False)

    for trace in fig.data:
        assert trace.x[-1] <= 2.0 + 1e-9, f"{tipo}: el grafico excede stop=2.0 s"


# ---------------------------------------------------------------------------
# Media +- desvio estandar (a traves del PlotEngine).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tipo, builder, ch_names", SIGNAL_CASES)
def test_media_y_desvio_genera_figura(tipo, builder, ch_names):
    """
    El grafico de media +- desvio debe generar una figura valida con
    tres trazos: banda inferior, banda superior y media.
    """
    signal = builder()

    engine = PlotEngine(
        data=signal.data,
        sfreq=signal.sfreq,
        ch_names=signal.info.ch_names,
    )

    fig = engine.plot_mean_std(start=0.0, stop=2.0)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3


# ---------------------------------------------------------------------------
# Anotaciones: deben dibujarse sobre el grafico, sin importar el tipo.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tipo, builder, ch_names", SIGNAL_CASES)
def test_anotaciones_se_dibujan_como_shapes(tipo, builder, ch_names):
    """
    Si la senal tiene anotaciones con duracion, deben aparecer como
    rectangulos/lineas (shapes) sobre la figura.
    """
    signal = builder()

    anotaciones = Anotaciones()
    anotaciones.add(onset=1.0, duration=0.5, description="Artefacto")
    anotaciones.add(onset=2.5, duration=0.0, description="Trigger")
    signal.set_anotaciones(anotaciones)

    fig = plot_raw(
        signal,
        show=False,
        show_annotations=True,
        fill_annotations=True,
    )

    # add_vrect y add_vline agregan formas a la figura.
    assert len(fig.layout.shapes) >= 2, (
        f"{tipo}: no se dibujaron las anotaciones sobre el grafico"
    )


@pytest.mark.parametrize("tipo, builder, ch_names", SIGNAL_CASES)
def test_sin_anotaciones_no_hay_shapes(tipo, builder, ch_names):
    """
    Una senal sin anotaciones no debe dibujar shapes de anotacion.
    """
    signal = builder()

    fig = plot_raw(signal, show=False, show_annotations=True)

    assert len(fig.layout.shapes) == 0


# ---------------------------------------------------------------------------
# Test cruzado: cada tipo grafica su propia cantidad de canales.
# ---------------------------------------------------------------------------


def test_cada_tipo_grafica_su_cantidad_de_canales():
    """
    Verifica de un vistazo que cada tipo de senal genera la cantidad
    de trazos correcta segun sus canales.
    """
    esperado = {
        "EEG": 3,
        "ECG": 2,
        "EMG": 2,
    }

    figuras = {
        "EEG": plot_raw(_make_eeg(), show=False),
        "ECG": plot_raw(_make_ecg(), show=False),
        "EMG": plot_raw(_make_emg(), show=False),
    }

    for tipo, fig in figuras.items():
        assert len(fig.data) == esperado[tipo], (
            f"{tipo}: se esperaban {esperado[tipo]} trazos y hay {len(fig.data)}"
        )
