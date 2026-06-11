import pytest

from tpi_analisisdesenales.eventos import Anotaciones


def test_anotaciones_add_and_find():
    anot = Anotaciones()

    anot.add(
        onset=10,
        duration=2,
        description="Artefacto ocular",
    )

    anot.add(
        onset=20,
        duration=1,
        description="Movimiento muscular",
    )

    assert len(anot) == 2

    results = anot.find("muscular")

    assert len(results) == 1
    assert results[0]["description"] == "Movimiento muscular"


def test_anotaciones_add_annotation_alias():
    anot = Anotaciones()

    anot.add_annotation(
        onset=5,
        duration=1.5,
        description="Parpadeo",
    )

    assert len(anot) == 1
    assert anot[0]["onset"] == 5.0
    assert anot[0]["duration"] == 1.5
    assert anot[0]["description"] == "Parpadeo"


def test_anotaciones_find_case_insensitive():
    anot = Anotaciones()

    anot.add(
        onset=3,
        duration=0.5,
        description="Artefacto Ocular",
    )

    results = anot.find("ocular")

    assert len(results) == 1
    assert results[0]["description"] == "Artefacto Ocular"


def test_anotaciones_get_annotations_dataframe():
    anot = Anotaciones()

    anot.add(
        onset=1,
        duration=0,
        description="Evento puntual",
    )

    df = anot.get_annotations()

    assert list(df.columns) == ["onset", "duration", "description"]
    assert len(df) == 1
    assert df.iloc[0]["onset"] == 1.0
    assert df.iloc[0]["duration"] == 0.0
    assert df.iloc[0]["description"] == "Evento puntual"


def test_anotaciones_to_list():
    anot = Anotaciones()

    anot.add(
        onset=2,
        duration=1,
        description="Ruido",
    )

    data = anot.to_list()

    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["description"] == "Ruido"


def test_anotaciones_clear():
    anot = Anotaciones()

    anot.add(
        onset=2,
        duration=1,
        description="Ruido",
    )

    assert len(anot) == 1

    anot.clear()

    assert len(anot) == 0


def test_anotaciones_onset_negativo_error():
    anot = Anotaciones()

    with pytest.raises(ValueError):
        anot.add(
            onset=-1,
            duration=1,
            description="Error",
        )


def test_anotaciones_duration_negativa_error():
    anot = Anotaciones()

    with pytest.raises(ValueError):
        anot.add(
            onset=1,
            duration=-1,
            description="Error",
        )