"""
Modulo para graficar objetos RawSignal usando PlotEngine.
"""

from typing import Optional, Sequence

from .plot_engine import PlotEngine


def plot_raw(
    raw,
    picks: Optional[Sequence[int | str] | int | str] = None,
    start: float = 0.0,
    stop: Optional[float] = None,
    superpose: bool = False,
    show_annotations: bool = True,
    fill_annotations: bool = True,
    title: str = "Visualizacion de senal continua",
):
    """
    Grafica un objeto RawSignal usando el motor grafico.

    Parameters
    ----------
    raw : RawSignal
        Objeto que contiene la senal continua.

    picks : int | str | list[int] | list[str] | None
        Canales a graficar. Si es None, se grafican todos.

    start : float
        Tiempo inicial del tramo a visualizar, en segundos.

    stop : float | None
        Tiempo final del tramo a visualizar, en segundos.
        Si es None, se grafica hasta el final.

    superpose : bool
        Si es True, superpone los canales.
        Si es False, los muestra separados.

    show_annotations : bool
        Indica si se muestran las anotaciones.

    fill_annotations : bool
        Indica si las anotaciones se muestran como zonas sombreadas.

    title : str
        Titulo del grafico.

    Returns
    -------
    fig
        Figura generada por PlotEngine.
    """

    annotations = _prepare_annotations(raw)

    engine = PlotEngine(
        data=raw.data,
        sfreq=raw.sfreq,
        ch_names=raw.info.ch_names,
        anotaciones=annotations,
    )

    fig = engine.plot_signals(
        picks=picks,
        start=start,
        stop=stop,
        superpose=superpose,
        show_annotations=show_annotations,
        fill_annotations=fill_annotations,
        title=title,
    )

    return fig


def _prepare_annotations(raw):
    """
    Convierte las anotaciones de RawSignal al formato esperado por PlotEngine.

    PlotEngine espera una lista de diccionarios con:
    onset, duration y description.
    """

    if raw.anotaciones is None or len(raw.anotaciones) == 0:
        return []

    df = raw.anotaciones.get_annotations()

    annotations = []

    for _, row in df.iterrows():
        annotations.append(
            {
                "onset": float(row["onset"]),
                "duration": float(row["duration"]),
                "description": str(row["description"]),
            }
        )

    return annotations