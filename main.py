import numpy as np

from tpi_analisisdesenales.info import Info
from tpi_analisisdesenales.raw_signal import RawSignal
from tpi_analisisdesenales.eventos import Eventos
from tpi_analisisdesenales.anotaciones import Anotaciones
from tpi_analisisdesenales.epocas import Epocas


def main():
    # Crear metadata de la señal
    info = Info(
        ch_names=["C3", "C4"],
        sfreq=250.0,
        ch_types=["eeg", "eeg"],
        subject_info={"id": "S01"},
    )

    # Crear datos simulados: 2 canales y 1000 muestras
    data = np.random.rand(2, 1000)

    # Crear objeto RawSignal
    raw = RawSignal(data=data, info=info)

    # Mostrar resumen general
    print(raw)
    print("Descripcion estadistica:")
    print(raw.describe())

    # Crear eventos del experimento
    ev = Eventos()
    ev.add(sample=200, event_id=1)
    ev.add(sample=600, event_id=2)
    print("Eventos:")
    print(ev.get_events())

    # Crear anotaciones asociadas al registro
    anot = Anotaciones(
        onset=[],
        duration=[],
        description=[],
    )
    anot.add(onset=10.0, duration=2.0, description="Artefacto ocular")
    anot.add(onset=25.0, duration=1.5, description="Movimiento muscular")

    print("Anotaciones:")
    print(anot.get_annotations())

    # Asociar anotaciones a la señal si RawSignal ya tiene ese metodo
    raw.set_anotaciones(anot)

    # Crear epocas a partir de los eventos
    epochs = Epocas(signal=raw, eventos=ev, tmin=-0.1, tmax=0.4)

    print(epochs)
    print("Forma de datos de epocas:", epochs.get_data().shape)
    print("Promedio de epocas:", epochs.average().shape)


if __name__ == "__main__":
    main()