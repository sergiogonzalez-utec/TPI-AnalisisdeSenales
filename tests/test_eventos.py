from tpi_analisisdesenales.eventos import Eventos


def test_eventos_add_event():
    ev = Eventos(sfreq=250.0)

    ev.add_event(
        onset=1.0,
        event_id=1,
        description="Estimulo A",
    )

    ev.add_event(
        onset=2.0,
        event_id=2,
        description="Estimulo B",
    )

    assert len(ev) == 2

    eventos = ev.get_events()

    assert eventos.iloc[0]["onset"] == 1.0
    assert eventos.iloc[0]["sample"] == 250
    assert eventos.iloc[0]["event_id"] == 1
    assert eventos.iloc[0]["description"] == "Estimulo A"

    assert eventos.iloc[1]["onset"] == 2.0
    assert eventos.iloc[1]["sample"] == 500
    assert eventos.iloc[1]["event_id"] == 2
    assert eventos.iloc[1]["description"] == "Estimulo B"


def test_eventos_rename_event_id():
    ev = Eventos(sfreq=250.0)

    ev.add_event(
        onset=1.0,
        event_id=1,
        description="Estimulo A",
    )

    ev.add_event(
        onset=2.0,
        event_id=2,
        description="Estimulo B",
    )

    ev.rename_event_id(
        old_id=2,
        new_id=99,
    )

    eventos = ev.get_events()

    assert eventos.iloc[0]["event_id"] == 1
    assert eventos.iloc[1]["event_id"] == 99


def test_eventos_to_list():
    ev = Eventos(sfreq=250.0)

    ev.add_event(
        onset=1.0,
        event_id=1,
        description="Estimulo",
    )

    data = ev.to_list()

    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["onset"] == 1.0
    assert data[0]["sample"] == 250
    assert data[0]["event_id"] == 1
    assert data[0]["description"] == "Estimulo"


def test_eventos_clear():
    ev = Eventos(sfreq=250.0)

    ev.add_event(
        onset=1.0,
        event_id=1,
        description="Estimulo",
    )

    assert len(ev) == 1

    ev.clear()

    assert len(ev) == 0


def test_eventos_onset_negativo_error():
    ev = Eventos(sfreq=250.0)

    try:
        ev.add_event(
            onset=-1.0,
            event_id=1,
            description="Error",
        )
        assert False
    except ValueError:
        assert True