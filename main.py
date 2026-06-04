import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tpi_analisisdesenales.preprocesamiento.filtros import (
    filtro_notch,
    filtro_pasaaltos,
    filtro_pasabajos,
)


def leer_openbci_txt(ruta, sfreq=250):
    df = pd.read_csv(
        ruta,
        header=None,
        comment="%",
        skip_blank_lines=True
    )

    # Convertir todas las columnas posibles a numero.
    # Lo que no pueda convertirse, como "EXG Channel 0", queda como NaN.
    df = df.apply(pd.to_numeric, errors="coerce")

    # Eliminar filas que no tengan numero de muestra.
    # Esto borra encabezados de texto.
    df = df.dropna(subset=[0])

    # Tomar columnas de senal biologica.
    data = df.iloc[:, 1:5].to_numpy(dtype=float)

    # Pasar de n_muestras x n_canales a n_canales x n_muestras.
    data = data.T

    return data, sfreq


def limpiar_senal(data, sfreq):
    data_limpia = filtro_notch(data, sfreq, freq=50.0, q=30.0)
    data_limpia = filtro_pasaaltos(data_limpia, sfreq, l_freq=0.5, order=4)
    data_limpia = filtro_pasabajos(data_limpia, sfreq, h_freq=40.0, order=4)

    return data_limpia


def crear_fig_senal_filtrada(data_limpia, sfreq, segundos=50):
    """
    Crea una figura Plotly con todos los canales filtrados,
    separados verticalmente y mostrando solo una ventana de tiempo.
    """

    n_canales, n_muestras = data_limpia.shape

    muestras_a_mostrar = int(segundos * sfreq)
    muestras_a_mostrar = min(muestras_a_mostrar, n_muestras)

    data_plot = data_limpia[:, :muestras_a_mostrar]
    tiempo = np.arange(muestras_a_mostrar) / sfreq

    separacion = np.nanmax(np.abs(data_plot)) * 2

    if separacion == 0:
        separacion = 1

    fig = go.Figure()

    for canal in range(n_canales):
        offset = canal * separacion
        senal = data_plot[canal] + offset

        fig.add_trace(
            go.Scattergl(
                x=tiempo,
                y=senal,
                mode="lines",
                name=f"Canal {canal + 1}"
            )
        )

    fig.update_layout(
        title=f"Senales filtradas - primeros {segundos} segundos",
        xaxis_title="Tiempo [s]",
        yaxis_title="Canales separados",
        template="plotly_white",
        height=700
    )

    return fig
def main():
    ruta = "docs\senal.txt"

    data, sfreq = leer_openbci_txt(ruta, sfreq=250)

    data_limpia = limpiar_senal(data, sfreq)

    fig = crear_fig_senal_filtrada(data_limpia, sfreq, segundos=50)

    fig.show()


if __name__ == "__main__":
    main()