class Anotaciones:
    def __init__(self):
        """
        Clase Anotaciones: almacena anotaciones sobre la señal.
        Cada anotación tiene onset (inicio), duración y descripción.
        """
        self.annotations = []

    def add(self, onset: float, duration: float, description: str, ch_names=None):
        """Agrega una nueva anotación."""
        self.annotations.append({
            "onset": onset,
            "duration": duration,
            "description": description,
            "ch_names": ch_names or []
        })

    def remove(self, index: int):
        """Elimina una anotación por índice."""
        if 0 <= index < len(self.annotations):
            self.annotations.pop(index)

    def get_annotations(self):
        """Devuelve todas las anotaciones."""
        return self.annotations

    def find(self, keyword: str):
        """Busca anotaciones que contengan una palabra clave en la descripción."""
        return [a for a in self.annotations if keyword.lower() in a["description"].lower()]

    def __len__(self):
        return len(self.annotations)

    def __iter__(self):
        return iter(self.annotations)

    def __str__(self):
        return f"Anotaciones: {len(self.annotations)} registradas"
