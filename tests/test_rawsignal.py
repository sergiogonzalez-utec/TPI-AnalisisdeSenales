import numpy as np

from tpi_analisisdesenales.info import Info
from tpi_analisisdesenales.signals import RawSignal


def test_rawsignal_describe_and_crop():
    info = Info(
        ch_names=["C3", "C4"],
        sfreq=250.0,
        ch_types=["EEG", "EEG"],
    )

    data = np.random.rand(2, 100)

    raw = RawSignal(
        data=data,
        info=info,
    )

    desc = raw.describe()

    assert desc.shape[0] == 2
    assert list(desc["name"]) == ["C3", "C4"]
    assert list(desc["type"]) == ["EEG", "EEG"]

    assert "min" in desc.columns
    assert "max" in desc.columns
    assert "rms" in desc.columns
    assert "var" in desc.columns

    raw.crop(0, 0.2)

    assert raw.get_data().shape == (2, 50)


def test_rawsignal_get_data():
    info = Info(
        ch_names=["C3", "C4"],
        sfreq=250.0,
        ch_types=["EEG", "EEG"],
    )

    data = np.random.rand(2, 100)

    raw = RawSignal(
        data=data,
        info=info,
    )

    output = raw.get_data()

    assert output.shape == (2, 100)
    assert np.allclose(output, data)


def test_rawsignal_crea_anotaciones_y_eventos_por_defecto():
    info = Info(
        ch_names=["C3", "C4"],
        sfreq=250.0,
        ch_types=["EEG", "EEG"],
    )

    data = np.random.rand(2, 100)

    raw = RawSignal(
        data=data,
        info=info,
    )

    assert raw.anotaciones is not None
    assert raw.eventos is not None
    assert len(raw.anotaciones) == 0
    assert len(raw.eventos) == 0