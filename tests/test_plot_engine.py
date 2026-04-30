import numpy as np
import pytest

from tpi_analisisdesenales.visualizacion import PlotEngine


class DummyAnotaciones:
    """
    Clase simple de apoyo para simular anotaciones en los tests.
    La usamos para no depender todavia de la clase real Anotaciones.
    """

    def __iter__(self):
        yield {
            "onset": 1.0,
            "duration": 0.5,
            "description": "Evento_1",
        }
        yield {
            "onset": 2.5,
            "duration": 0.2,
            "description": "Evento_2",
        }


def test_plot_engine_init_ok():
    """
    Verifica que PlotEngine se construya correctamente
    y que genere bien el vector temporal.
    """
    data = np.random.randn(2, 1000)
    sfreq = 250
    ch_names = ["C3", "C4"]

    engine = PlotEngine(data=data, sfreq=sfreq, ch_names=ch_names)

    assert engine.data.shape == (2, 1000)
    assert engine.sfreq == 250.0
    assert engine.ch_names == ["C3", "C4"]
    assert len(engine.times) == 1000
    assert engine.times[0] == 0.0


def test_plot_engine_invalid_data_dimension():
    """
    Verifica que falle si data no es 2D.
    """
    data = np.random.randn(1000)
    sfreq = 250
    ch_names = ["C3"]

    with pytest.raises(ValueError, match="data debe tener forma"):
        PlotEngine(data=data, sfreq=sfreq, ch_names=ch_names)


def test_plot_engine_invalid_sfreq():
    """
    Verifica que falle si la frecuencia de muestreo no es valida.
    """
    data = np.random.randn(2, 1000)
    ch_names = ["C3", "C4"]

    with pytest.raises(ValueError, match="sfreq debe ser mayor que cero"):
        PlotEngine(data=data, sfreq=0, ch_names=ch_names)


def test_plot_engine_invalid_channel_names_length():
    """
    Verifica que falle si la cantidad de nombres no coincide con los canales.
    """
    data = np.random.randn(2, 1000)
    sfreq = 250
    ch_names = ["C3"]

    with pytest.raises(ValueError, match="cantidad de nombres de canal"):
        PlotEngine(data=data, sfreq=sfreq, ch_names=ch_names)


def test_resolve_picks_with_string():
    """
    Verifica seleccion de un canal por nombre.
    """
    data = np.random.randn(2, 1000)
    engine = PlotEngine(data=data, sfreq=250, ch_names=["C3", "C4"])

    indices = engine._resolve_picks("C4")

    assert indices == [1]


def test_resolve_picks_with_list():
    """
    Verifica seleccion multiple de canales.
    """
    data = np.random.randn(3, 1000)
    engine = PlotEngine(data=data, sfreq=250, ch_names=["C3", "Cz", "C4"])

    indices = engine._resolve_picks(["C3", "C4"])

    assert indices == [0, 2]


def test_resolve_picks_invalid_channel():
    """
    Verifica que falle si se pide un canal inexistente.
    """
    data = np.random.randn(2, 1000)
    engine = PlotEngine(data=data, sfreq=250, ch_names=["C3", "C4"])

    with pytest.raises(ValueError, match="no existe"):
        engine._resolve_picks("Cz")


def test_time_to_sample_ok():
    """
    Verifica conversion correcta de tiempo a indices de muestra.
    """
    data = np.random.randn(2, 1000)
    engine = PlotEngine(data=data, sfreq=250, ch_names=["C3", "C4"])

    idx_start, idx_stop = engine._time_to_sample(start=1.0, stop=2.0)

    assert idx_start == 250
    assert idx_stop == 500


def test_get_segment_ok():
    """
    Verifica que el recorte de datos tenga forma correcta.
    """
    data = np.random.randn(3, 1000)
    engine = PlotEngine(data=data, sfreq=250, ch_names=["C3", "Cz", "C4"])

    segment_data, segment_times, indices = engine._get_segment(
        picks=["C3", "C4"],
        start=0.0,
        stop=2.0,
    )

    assert segment_data.shape == (2, 500)
    assert len(segment_times) == 500
    assert indices == [0, 2]


def test_get_visible_annotations():
    """
    Verifica que solo se obtengan anotaciones visibles en el rango.
    """
    data = np.random.randn(2, 1000)
    anotaciones = DummyAnotaciones()

    engine = PlotEngine(
        data=data,
        sfreq=250,
        ch_names=["C3", "C4"],
        anotaciones=anotaciones,
    )

    visibles = engine._get_visible_annotations(start=0.5, stop=1.6)

    assert len(visibles) == 1
    assert visibles[0]["description"] == "Evento_1"


def test_plot_signals_returns_figure():
    """
    Verifica que plot_signals devuelva una figura de plotly.
    """
    data = np.random.randn(2, 1000)
    engine = PlotEngine(data=data, sfreq=250, ch_names=["C3", "C4"])

    fig = engine.plot_signals(start=0.0, stop=2.0)

    assert fig is not None
    assert hasattr(fig, "data")
    assert len(fig.data) == 2


def test_plot_signals_with_annotations():
    """
    Verifica que el grafico se genere correctamente aun con anotaciones.
    """
    data = np.random.randn(2, 1000)
    anotaciones = DummyAnotaciones()

    engine = PlotEngine(
        data=data,
        sfreq=250,
        ch_names=["C3", "C4"],
        anotaciones=anotaciones,
    )

    fig = engine.plot_signals(start=0.0, stop=3.0)

    assert fig is not None
    assert hasattr(fig, "layout")


def test_plot_mean_std_returns_figure():
    """
    Verifica que plot_mean_std devuelva una figura valida.
    """
    data = np.random.randn(3, 1000)
    engine = PlotEngine(data=data, sfreq=250, ch_names=["C3", "Cz", "C4"])

    fig = engine.plot_mean_std(start=0.0, stop=2.0)

    assert fig is not None
    assert hasattr(fig, "data")
    assert len(fig.data) == 3