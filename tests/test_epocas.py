import numpy as np

from tpi_analisisdesenales.info import Info
from tpi_analisisdesenales.raw_signal import RawSignal
from tpi_analisisdesenales.eventos import Eventos
from tpi_analisisdesenales.epocas import Epocas


def test_epocas_crea_segmentos_correctamente():
    sfreq = 100.0

    info = Info(
        ch_names=["C3", "C4"],
        sfreq=sfreq,
        ch_types=["EEG", "EEG"],
    )

    data = np.random.randn(2, 1000)

    eventos = Eventos(
        samples=[200, 500],
        event_id=[1, 1],
    )

    raw = RawSignal(
        data=data,
        info=info,
        eventos=eventos,
    )

    epocas = Epocas(
        raw=raw,
        eventos=eventos,
        event_id=1,
        tmin=-0.1,
        tmax=0.4,
    )

    datos = epocas.get_data()

    assert datos.shape == (2, 2, 50)