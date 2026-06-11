"""
Modulo para graficar objetos Epocas usando Plotly.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def plot_epochs(
    epocas,
    picks=None,
    max_epochs: int = 10,
    superpose: bool = False,
    normalize: bool = True,
    title: str = "Visualizacion de epocas",
    show: bool = True,
):
    """
    Grafica epocas individuales.

    Parameters
    ----------
    epocas : Epocas
        Objeto Epocas.

    picks : int | str | list[int] | list[str] | None
        Canales a graficar.

    max_epochs : int
        Cantidad maxima de epocas a mostrar.

    superpose : bool
        Si es True, superpone las epocas.
        Si es False, las separa verticalmente.

    normalize : bool
        Si es True, normaliza cada trazo solo para visualizacion.

    title : str
        Titulo del grafico.

    show : bool
        Si es True, muestra la figura.

    Returns
    -------
    fig
        Figura Plotly.
    """

    data = epocas.get_data()
    times = epocas.times
    ch_names = epocas.ch_names

    indices_canales = _resolver_picks(picks, ch_names)

    n_epocas = min(data.shape[0], max_epochs)

    fig = go.Figure()

    offset_step = 3.0
    yticks = []
    ylabels = []

    contador = 0

    for epoca_idx in range(n_epocas):
        for ch_idx in indices_canales:
            y = data[epoca_idx, ch_idx, :].astype(float)

            if normalize:
                y = y - np.nanmean(y)
                max_abs = np.nanmax(np.abs(y))

                if max_abs != 0:
                    y = y / max_abs

            if superpose:
                offset = 0.0
            else:
                offset = contador * offset_step
                yticks.append(offset)
                ylabels.append(f"E{epoca_idx + 1} - {ch_names[ch_idx]}")
                contador += 1

            fig.add_trace(
                go.Scattergl(
                    x=times,
                    y=y + offset,
                    mode="lines",
                    name=f"Epoca {epoca_idx + 1} - {ch_names[ch_idx]}",
                    line=dict(width=1.1),
                )
            )

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_width=1,
        annotation_text="Evento",
        annotation_position="top",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Tiempo relativo al evento [s]",
        yaxis_title="Epocas / canales" if not superpose else "Amplitud",
        template="plotly_white",
        hovermode="x unified",
        height=750,
        margin=dict(l=120, r=130, t=80, b=70),
        legend_title="Trazos",
    )

    if not superpose:
        fig.update_yaxes(
            tickmode="array",
            tickvals=yticks,
            ticktext=ylabels,
            showgrid=True,
            zeroline=False,
        )
    else:
        fig.update_yaxes(
            showgrid=True,
            zeroline=False,
        )

    fig.update_xaxes(
        showgrid=True,
        zeroline=True,
    )

    if show:
        fig.show()

    return fig


def plot_epochs_average(
    epocas,
    picks=None,
    normalize: bool = False,
    title: str = "Promedio de epocas",
    show: bool = True,
):
    """
    Grafica el promedio de las epocas.

    Parameters
    ----------
    epocas : Epocas
        Objeto Epocas.

    picks : int | str | list[int] | list[str] | None
        Canales a graficar.

    normalize : bool
        Si es True, normaliza cada canal solo para visualizacion.

    title : str
        Titulo del grafico.

    show : bool
        Si es True, muestra la figura.

    Returns
    -------
    fig
        Figura Plotly.
    """

    avg = epocas.average()
    times = epocas.times
    ch_names = epocas.ch_names

    indices_canales = _resolver_picks(picks, ch_names)

    fig = go.Figure()

    offset_step = 3.0
    yticks = []
    ylabels = []

    for i, ch_idx in enumerate(indices_canales):
        y = avg[ch_idx, :].astype(float)

        if normalize:
            y = y - np.nanmean(y)
            max_abs = np.nanmax(np.abs(y))

            if max_abs != 0:
                y = y / max_abs

        offset = i * offset_step

        yticks.append(offset)
        ylabels.append(ch_names[ch_idx])

        fig.add_trace(
            go.Scattergl(
                x=times,
                y=y + offset,
                mode="lines",
                name=ch_names[ch_idx],
                line=dict(width=1.5),
            )
        )

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_width=1,
        annotation_text="Evento",
        annotation_position="top",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Tiempo relativo al evento [s]",
        yaxis_title="Canales",
        template="plotly_white",
        hovermode="x unified",
        height=700,
        margin=dict(l=100, r=130, t=80, b=70),
        legend_title="Canales",
    )

    fig.update_yaxes(
        tickmode="array",
        tickvals=yticks,
        ticktext=ylabels,
        showgrid=True,
        zeroline=False,
    )

    fig.update_xaxes(
        showgrid=True,
        zeroline=True,
    )

    if show:
        fig.show()

    return fig


def _resolver_picks(picks, ch_names):
    """
    Convierte picks a indices de canales.
    """

    if picks is None:
        return list(range(len(ch_names)))

    if isinstance(picks, (int, str)):
        picks = [picks]

    indices = []

    for pick in picks:
        if isinstance(pick, int):
            indices.append(pick)

        elif isinstance(pick, str):
            if pick not in ch_names:
                raise ValueError(f"Canal no encontrado: {pick}")

            indices.append(ch_names.index(pick))

        else:
            raise TypeError("picks debe contener indices o nombres de canales.")

    return indices