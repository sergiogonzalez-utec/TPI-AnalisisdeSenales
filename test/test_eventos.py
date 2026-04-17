from src.eventos import Eventos

def test_eventos_add_and_rename():
    ev = Eventos()
    ev.add_event(sample=100, event_id=1)
    ev.add_event(sample=200, event_id=2)

    assert len(ev) == 2
    ev.rename_event_id(2, 99)
    assert ev.get_events()[1]["event_id"] == 99

