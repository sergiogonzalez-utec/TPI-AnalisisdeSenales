import numpy as np
import matplotlib.pyplot as plt

from tpi_analisisdesenales.info.info import Info
from tpi_analisisdesenales.signals.raw_signal import RawSignal
from tpi_analisisdesenales.eventos.eventos import Eventos
from tpi_analisisdesenales.eventos.anotaciones import Anotaciones
from tpi_analisisdesenales.epocas.epocas import Epocas


def main():
    print("Ejecutando main.py")

    info = Info(
        ch_names=["C3", "C4"],
        sfreq=100.0,
        ch_types=["EEG", "EEG"],
    )

    data = np.random.randn(2, 1000)

    eventos = Eventos(
        samples=[200, 500, 800],
        event_id=[1, 1, 2],
    )

    anotaciones = Anotaciones()

    anotaciones.add(
        onset=2.5,
        duration=1.0,
        description="artefacto",
        ch_names=["C3"],
    )

    raw = RawSignal(
        data=data,
        info=info,
        eventos=eventos,
        anotaciones=anotaciones,
    )

    print("Senal creada:")
    print(raw)

    print("\nEventos:")
    print(eventos.get_events())

    print("\nAnotaciones:")
    print(anotaciones.get_annotations())

    epocas = Epocas(
        raw=raw,
        eventos=eventos,
        event_id=1,
        tmin=-0.1,
        tmax=0.4,
    )

    print("\nEpocas creadas:")
    print(epocas)

    print("\nForma de los datos de epocas:")
    print(epocas.get_data().shape)

    fig = raw.plot()

    if fig is not None:
        fig.show()

if __name__ == "__main__":
    main()