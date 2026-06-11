import numpy as np
import pytest

from tpi_analisisdesenales.info import Info
from tpi_analisisdesenales.signals import RawSignal
from tpi_analisisdesenales.eventos import Eventos
from tpi_analisisdesenales.epocas import Epocas


def crear_raw_de_prueba():
    info = Info(
        ch_names=["C3", "C4"],
        sfreq=100.0,
        ch_types=["EEG", "EEG"],
    )

    data = np.arange(2 * 1000, dtype=float).reshape(2, 1000)

    raw = RawSignal(
        data=data,
        info=info,
    )

    return raw


def test_epocas_creacion_basica():
    raw = crear_raw_de_prueba()

    eventos = Eventos(sfreq=raw.sfreq)

    eventos.add_event(
        onset=2.0,
        event_id=1,
        description="Estimulo A",
    )

    eventos.add_event(
        onset=5.0,
        event_id=1,
        description="Estimulo A",
    )

    epocas = Epocas(
        raw=raw,
        eventos=eventos,
        tmin=-0.1,
        tmax=0.3,
    )

    data = epocas.get_data()

    assert data.shape == (2, 2, 40)
    assert len(epocas) == 2
    assert epocas.times.shape == (40,)
    assert epocas.times[0] == pytest.approx(-0.1)


def test_epocas_usa_eventos_del_raw():
    raw = crear_raw_de_prueba()

    raw.eventos.add_event(
        onset=2.0,
        event_id=1,
        description="Estimulo",
    )

    epocas = Epocas(
        raw=raw,
        tmin=-0.1,
        tmax=0.3,
    )

    assert epocas.get_data().shape == (1, 2, 40)


def test_epocas_average():
    raw = crear_raw_de_prueba()

    eventos = Eventos(sfreq=raw.sfreq)

    eventos.add_event(
        onset=2.0,
        event_id=1,
        description="Estimulo A",
    )

    eventos.add_event(
        onset=5.0,
        event_id=1,
        description="Estimulo A",
    )

    epocas = Epocas(
        raw=raw,
        eventos=eventos,
        tmin=-0.1,
        tmax=0.3,
    )

    promedio = epocas.average()

    assert promedio.shape == (2, 40)


def test_epocas_metadata():
    raw = crear_raw_de_prueba()

    eventos = Eventos(sfreq=raw.sfreq)

    eventos.add_event(
        onset=2.0,
        event_id=7,
        description="Estimulo especial",
    )

    epocas = Epocas(
        raw=raw,
        eventos=eventos,
        tmin=-0.1,
        tmax=0.3,
    )

    metadata = epocas.get_metadata()

    assert len(metadata) == 1
    assert metadata.iloc[0]["onset"] == 2.0
    assert metadata.iloc[0]["sample"] == 200
    assert metadata.iloc[0]["event_id"] == 7
    assert metadata.iloc[0]["description"] == "Estimulo especial"
    assert metadata.iloc[0]["start_sample"] == 190
    assert metadata.iloc[0]["stop_sample"] == 230


def test_epocas_filtra_event_id():
    raw = crear_raw_de_prueba()

    eventos = Eventos(sfreq=raw.sfreq)

    eventos.add_event(
        onset=2.0,
        event_id=1,
        description="Estimulo A",
    )

    eventos.add_event(
        onset=5.0,
        event_id=2,
        description="Estimulo B",
    )

    epocas = Epocas(
        raw=raw,
        eventos=eventos,
        tmin=-0.1,
        tmax=0.3,
        event_id=2,
    )

    metadata = epocas.get_metadata()

    assert len(epocas) == 1
    assert metadata.iloc[0]["event_id"] == 2
    assert metadata.iloc[0]["description"] == "Estimulo B"


def test_epocas_select_event_id():
    raw = crear_raw_de_prueba()

    eventos = Eventos(sfreq=raw.sfreq)

    eventos.add_event(
        onset=2.0,
        event_id=1,
        description="Estimulo A",
    )

    eventos.add_event(
        onset=5.0,
        event_id=2,
        description="Estimulo B",
    )

    epocas = Epocas(
        raw=raw,
        eventos=eventos,
        tmin=-0.1,
        tmax=0.3,
    )

    epocas_evento_2 = epocas.select_event_id(2)

    metadata = epocas_evento_2.get_metadata()

    assert len(epocas_evento_2) == 1
    assert metadata.iloc[0]["event_id"] == 2


def test_epocas_descarta_fuera_de_limites():
    raw = crear_raw_de_prueba()

    eventos = Eventos(sfreq=raw.sfreq)

    eventos.add_event(
        onset=0.05,
        event_id=1,
        description="Evento muy temprano",
    )

    eventos.add_event(
        onset=2.0,
        event_id=1,
        description="Evento valido",
    )

    epocas = Epocas(
        raw=raw,
        eventos=eventos,
        tmin=-0.1,
        tmax=0.3,
        reject_out_of_bounds=True,
    )

    metadata = epocas.get_metadata()

    assert len(epocas) == 1
    assert metadata.iloc[0]["description"] == "Evento valido"


def test_epocas_error_sin_eventos():
    raw = crear_raw_de_prueba()

    eventos = Eventos(sfreq=raw.sfreq)

    with pytest.raises(ValueError):
        Epocas(
            raw=raw,
            eventos=eventos,
            tmin=-0.1,
            tmax=0.3,
        )


def test_epocas_error_tmin_mayor_tmax():
    raw = crear_raw_de_prueba()

    eventos = Eventos(sfreq=raw.sfreq)

    eventos.add_event(
        onset=2.0,
        event_id=1,
        description="Estimulo",
    )

    with pytest.raises(ValueError):
        Epocas(
            raw=raw,
            eventos=eventos,
            tmin=0.5,
            tmax=0.1,
        )


def test_epocas_getitem():
    raw = crear_raw_de_prueba()

    eventos = Eventos(sfreq=raw.sfreq)

    eventos.add_event(
        onset=2.0,
        event_id=1,
        description="Estimulo",
    )

    epocas = Epocas(
        raw=raw,
        eventos=eventos,
        tmin=-0.1,
        tmax=0.3,
    )

    primera_epoca = epocas[0]

    assert primera_epoca.shape == (2, 40)