from tpi_analisisdesenales.info import Info

def test_info_basic():
    info = Info(ch_names=["C3", "C4"], sfreq=250.0, subject_info={"id": "S01"})
    assert info.n_channels() == 2
    assert info.sfreq == 250.0
    assert info.subject_info["id"] == "S01"
