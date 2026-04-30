import numpy as np
from tpi_analisisdesenales.info import Info
from tpi_analisisdesenales.raw_signal import RawSignal

def test_rawsignal_describe_and_crop():
    info = Info(ch_names=["C3", "C4"], sfreq=250.0)
    data = np.random.rand(2, 100)
    raw = RawSignal(data=data, sfreq=250.0, info=info)

    desc = raw.describe()
    assert desc["channels"] == 2
    assert desc["samples"] == 100

    raw.crop(0, 50)
    assert raw.get_data().shape == (2, 50)
