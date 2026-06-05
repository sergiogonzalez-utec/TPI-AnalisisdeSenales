from pathlib import Path

import pandas as pd

from tpi_analisisdesenales.info.info import Info
from tpi_analisisdesenales.signals.raw_signal import RawSignal

from tpi_analisisdesenales.preprocesamiento.filtros import (
    filtro_notch,
    filtro_pasaaltos,
    filtro_pasabajos,
)

from tpi_analisisdesenales.visualizacion.plot_raw import plot_raw


def leer_openbci_txt(ruta, sfreq=250):
    """
    Lee un archivo TXT de OpenBCI para probar la libreria.

    Devuelve un objeto RawSignal.
    """

    ruta = Path(ruta)

    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {ruta}")

    df = pd.read_csv(
        ruta,
        header=None,
        comment="%",
        skip_blank_lines=True,
    )

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=[0])

    data = df.iloc[:, 1:5].to_numpy(dtype=float)
    data = data.T

    info = Info(
        sfreq=sfreq,
        ch_names=["Canal 1", "Canal 2", "Canal 3", "Canal 4"],
        ch_types=["EEG", "EEG", "EEG", "EEG"],
    )

    raw = RawSignal(
        data=data,
        info=info,
    )

    return raw


def limpiar_raw(raw):
    """
    Aplica una cadena basica de limpieza a un objeto RawSignal.
    """

    data_limpia = filtro_notch(
        raw.data,
        raw.sfreq,
        freq=50.0,
        q=30.0,
    )

    data_limpia = filtro_pasaaltos(
        data_limpia,
        raw.sfreq,
        l_freq=0.5,
        order=4,
    )

    data_limpia = filtro_pasabajos(
        data_limpia,
        raw.sfreq,
        h_freq=40.0,
        order=4,
    )

    raw_limpia = RawSignal(
        data=data_limpia,
        info=raw.info,
    )

    return raw_limpia


def main():
    ruta = "docs/senal.txt"

    raw = leer_openbci_txt(
        ruta=ruta,
        sfreq=250,
    )

    raw_limpia = limpiar_raw(raw)

    plot_raw(
        raw_limpia,
        start=0,
        stop=50,
        superpose=False,
        show_annotations=True,
        fill_annotations=True,
        normalize=True,
        title="Senales filtradas",
        show=True,
    )


if __name__ == "__main__":
    main()