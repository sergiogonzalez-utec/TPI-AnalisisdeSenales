from tpi_analisisdesenales.info import Info


def test_info_basic():
    info = Info(
        ch_names=["C3", "C4"],
        sfreq=250.0,
        ch_types=["EEG", "EEG"],
        subject_info={"id": "S01"},
    )

    assert info.n_channels == 2
    assert info.sfreq == 250.0
    assert info.subject_info["id"] == "S01"


def test_info_channel_names():
    info = Info(
        ch_names=["C3", "C4", "Pz"],
        sfreq=250.0,
        ch_types=["EEG", "EEG", "EEG"],
    )

    assert info.ch_names == ["C3", "C4", "Pz"]
    assert info.n_channels == 3


def test_info_channel_types():
    info = Info(
        ch_names=["C3", "C4"],
        sfreq=250.0,
        ch_types=["EEG", "EEG"],
    )

    assert info.ch_types == ["EEG", "EEG"]


def test_info_default_subject_info():
    info = Info(
        ch_names=["C3", "C4"],
        sfreq=250.0,
        ch_types=["EEG", "EEG"],
    )

    assert info.subject_info is not None
    assert isinstance(info.subject_info, dict)


def test_info_repr():
    info = Info(
        ch_names=["C3", "C4"],
        sfreq=250.0,
        ch_types=["EEG", "EEG"],
    )

    texto = repr(info)

    assert "Info" in texto