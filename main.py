import numpy as np

from tpi_analisisdesenales.info import Info
from tpi_analisisdesenales.anotaciones import Anotaciones
from tpi_analisisdesenales.eventos import Eventos
from tpi_analisisdesenales.visualizacion import PlotEngine


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

    # Crear eventos del experimento
    ev = Eventos()
    ev.add(sample=200, event_id=1)
    ev.add(sample=600, event_id=2)

    print("Eventos:")
    print(ev.get_events())

    # Crear anotaciones dentro del rango visible de la señal
    anot = Anotaciones(
        onset=[],
        duration=[],
        description=[],
    )
    anot.add(onset=1.0, duration=0.5, description="Artefacto ocular")
    anot.add(onset=2.5, duration=0.4, description="Movimiento muscular")

    print("Anotaciones:")
    print(anot.get_annotations())

    # Crear motor grafico directamente
    engine = PlotEngine(
        data=data,
        sfreq=info.sfreq,
        ch_names=info.ch_names,
        anotaciones=anot,
    )

    # Mostrar grafico de señales
    fig = engine.plot_signals(
        start=0.0,
        stop=4.0,
        superpose=False,
        show_annotations=True,
        fill_annotations=True,
        title="Señal cruda con anotaciones",
    )
    fig.show()

    # Mostrar grafico de media y desvio
    fig2 = engine.plot_mean_std(
        start=0.0,
        stop=4.0,
        title="Media y desvio estandar",
    )
    fig2.show()


if __name__ == "__main__":
    main()