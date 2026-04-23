class Eventos:
    def __init__(self):
        """
        Clase Eventos: almacena eventos asociados a la señal.
        Cada evento tiene un sample (índice de muestra) y un event_id.
        """
        self.events = []  # lista de diccionarios con {"sample": int, "event_id": int}

    def add_event(self, sample: int, event_id: int):
        """Agrega un nuevo evento."""
        self.events.append({"sample": sample, "event_id": event_id})

    def remove_event(self, index: int):
        """Elimina un evento por índice."""
        if 0 <= index < len(self.events):
            self.events.pop(index)

    def get_events(self):
        """Devuelve todos los eventos."""
        return self.events

    def find_event(self, event_id: int):
        """Busca eventos por su ID."""
        return [e for e in self.events if e["event_id"] == event_id]

    def rename_event_id(self, old_id: int, new_id: int):
        """Renombra un event_id existente."""
        for e in self.events:
            if e["event_id"] == old_id:
                e["event_id"] = new_id

    def __len__(self):
        return len(self.events)

    def __str__(self):
        return f"Eventos: {len(self.events)} registrados"
