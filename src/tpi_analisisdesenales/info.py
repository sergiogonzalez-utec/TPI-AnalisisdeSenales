class Info:
    def __init__(self, ch_names, sfreq, ch_types=None, description="", experimenter="", subject_info=None, filtros=None, duration=None):
        """
        Clase Info: almacena metadatos de la señal.

        Parameters
        ----------
        ch_names : list
            Lista con los nombres de los canales.
        sfreq : float
            Frecuencia de muestreo en Hz.
        ch_types : list, optional
            Tipos de canales (ej. 'EEG', 'ECG', 'EMG').
        description : str, optional
            Descripción general de la señal.
        experimenter : str, optional
            Nombre del experimentador.
        subject_info : dict, optional
            Información del sujeto (ej. edad, género, ID).
        filtros : dict, optional
            Información de filtros aplicados.
        duration : float, optional
            Duración de la señal en segundos.
        """
        self.ch_names = ch_names
        self.sfreq = sfreq
        self.ch_types = ch_types or []
        self.description = description
        self.experimenter = experimenter
        self.subject_info = subject_info or {}
        self.filtros = filtros or {}
        self.duration = duration

    # Métodos útiles
    def n_channels(self):
        return len(self.ch_names)

    def n_samples(self, total_samples):
        """Devuelve el número de muestras totales (se pasa desde RawSignal)."""
        return total_samples

    def rename_channels(self, new_names):
        """Renombra los canales según una lista de nuevos nombres."""
        if len(new_names) == len(self.ch_names):
            self.ch_names = new_names
        else:
            raise ValueError("La cantidad de nuevos nombres no coincide con la cantidad de canales.")

    def add_bad(self, ch_name):
        """Marca un canal como 'malo'."""
        if not hasattr(self, "bads"):
            self.bads = []
        self.bads.append(ch_name)

    def remove_bad(self, ch_name):
        """Elimina un canal de la lista de 'malos'."""
        if hasattr(self, "bads") and ch_name in self.bads:
            self.bads.remove(ch_name)

    def __str__(self):
        return f"Info: {self.n_channels()} canales, sfreq={self.sfreq} Hz, sujeto={self.subject_info}"
