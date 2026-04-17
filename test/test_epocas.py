import numpy as np
from src.info import Info
from src.raw_signal import RawSignal
from src.eventos import Eventos
from src.epocas import Epocas

def test_epocas_extraction_and_average():
    info = Info(ch_names=["C3", "C4"], sfreq=250.0)
    data = np.random.rand(2, 1000)
    raw = RawSignal(data=data, sfreq=250.0, info=info)

    ev = Eventos()
    ev.add_event(sample=200, event_id=1)
    ev.add_event(sample=600, event_id=2)

    epochs = Epocas(signal=raw, eventos=ev, tmin=-0.1, tmax=0.4)
    assert len(epochs) == 2
    avg = epochs.average()
    assert avg.shape[0] == 2  # canales
