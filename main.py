import numpy as np
from src.info import Info
from src.raw_signal import RawSignal
from src.eventos import Eventos
from src.anotaciones import Anotaciones
from src.epocas import Epocas

def main():
    # 1. Crear Info
    info = Info(ch_names=["C3", "C4"], sfreq=250.0, subject_info={"id": "S01"})

    # 2. Crear datos simulados (2 canales, 1000 muestras)
    data = np.random.rand(2, 1000)
    raw = RawSignal(data=data, sfreq=250.0, info=info)

    print(raw)  # resumen de la señal
    print("Descripción:", raw.describe())

    # 3. Agregar eventos
    ev = Eventos()
    ev.add_event(sample=200, event_id=1)
    ev.add_event(sample=600, event_id=2)
    print("Eventos:", ev.get_events())

    # 4. Agregar anotaciones
    anot = Anotaciones()
    anot.add(onset=10, duration=2, description="Artefacto ocular")
    anot.add(onset=25, duration=1.5, description="Movimiento muscular")
    print("Anotaciones:", anot.get_annotations())

    # 5. Crear Epocas
    epochs = Epocas(signal=raw, eventos=ev, tmin=-0.1, tmax=0.4)
    print(epochs)
    print("Forma de datos de épocas:", epochs.get_data().shape)
    print("Promedio de épocas:", epochs.average().shape)

if __name__ == "__main__":
    main()
