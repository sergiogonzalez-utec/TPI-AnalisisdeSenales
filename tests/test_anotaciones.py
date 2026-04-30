from tpi_analisisdesenales.anotaciones import Anotaciones

def test_anotaciones_add_and_find():
    anot = Anotaciones()
    anot.add(onset=10, duration=2, description="Artefacto ocular")
    anot.add(onset=20, duration=1, description="Movimiento muscular")

    assert len(anot) == 2
    results = anot.find("muscular")
    assert len(results) == 1
    assert results[0]["description"] == "Movimiento muscular"
