class Info:
    """
    Clase Info

    Esta clase almacena los metadatos asociados a una señal fisiologica.
    No guarda la señal en si, sino informacion descriptiva que luego
    usaran otras clases como RawSignal, EEGSignal, ECGSignal o EMGSignal.

    Ejemplos de metadata:
    - nombres de canales
    - tipos de canales
    - frecuencia de muestreo
    - canales marcados como malos
    - informacion del experimento
    - filtros aplicados
    - cantidad de muestras
    - duracion total
    """

    def __init__(
        self,
        ch_names,
        sfreq,
        ch_types=None,
        bads=None,
        description="",
        experimenter="",
        subject_info=None,
        line_freq=None,
        lowpass=None,
        highpass=None,
        notch_freqs=None,
        n_samples=None,
    ):
        """
        Inicializa el objeto Info con la metadata basica.

        Parameters
        ----------
        ch_names : list
            Lista con los nombres de los canales.
        sfreq : int | float
            Frecuencia de muestreo en Hz.
        ch_types : list, optional
            Lista con el tipo de cada canal.
            Ejemplo: ["eeg", "eeg", "ecg"]
        bads : list, optional
            Lista de canales marcados como malos.
        description : str, optional
            Descripcion general de la señal o experimento.
        experimenter : str, optional
            Nombre del experimentador.
        subject_info : dict, optional
            Informacion adicional del sujeto.
        line_freq : int | float, optional
            Frecuencia de linea electrica, por ejemplo 50 o 60 Hz.
        lowpass : int | float, optional
            Frecuencia de corte superior del filtro aplicado durante el registro.
        highpass : int | float, optional
            Frecuencia de corte inferior del filtro aplicado durante el registro.
        notch_freqs : list, optional
            Frecuencias notch aplicadas, por ejemplo [50] o [50, 100].
        n_samples : int, optional
            Cantidad total de muestras de la señal.
            Este valor puede quedar en None al inicio y luego ser completado
            desde RawSignal cuando se cargue el archivo de datos.
        """
#-----------VALIDACIONES INICIALES----------------


        # Validamos que ch_names sea una lista y que no este vacia.
        # Esto es importante porque casi todas las operaciones futuras
        # dependen de saber que canales existen.
        if not isinstance(ch_names, list) or len(ch_names) == 0:
            raise ValueError("ch_names debe ser una lista no vacia.")

        # Validamos que la frecuencia de muestreo sea numerica y positiva.
        # Sin una sfreq valida no se puede calcular ni tiempo ni duracion.
        if not isinstance(sfreq, (int, float)) or sfreq <= 0:
            raise ValueError("sfreq debe ser un numero positivo.")

        # Si no se pasan tipos de canal, asignamos un tipo generico a todos.
        # Esto deja la clase util aunque el usuario no haya definido todo.
        if ch_types is None:
            ch_types = ["misc"] * len(ch_names)

        # Validamos que haya un tipo por cada canal.
        if len(ch_types) != len(ch_names):
            raise ValueError("ch_types debe tener la misma longitud que ch_names.")

#------------FIN DE VALIDACIONES----------------


#-----------GUARDADO DE ATRIBUTOS----------------
        # Guardamos internamente los atributos principales.


        # Se usan con guion bajo para indicar que son internos
        # y luego acceder a ellos mediante propiedades.
        self._ch_names = list(ch_names)
        self._sfreq = float(sfreq)
        self._ch_types = list(ch_types)
        self._bads = list(bads) if bads is not None else []

        # Metadatos generales del experimento o del sujeto.
        self.description = description
        self.experimenter = experimenter
        self.subject_info = subject_info if subject_info is not None else {}

        # Informacion relacionada con filtrado o condiciones de registro.
        self.line_freq = line_freq
        self.lowpass = lowpass
        self.highpass = highpass
        self.notch_freqs = list(notch_freqs) if notch_freqs is not None else []

        # Cantidad de muestras.
        # Puede venir desde afuera o completarse luego al cargar la señal.
        self._n_samples = None
        self.n_samples = n_samples

    @property
    def ch_names(self):
        """
        Devuelve la lista de nombres de canales.
        """
        return self._ch_names

    @property
    def sfreq(self):
        """
        Devuelve la frecuencia de muestreo en Hz.
        """
        return self._sfreq

    @property
    def ch_types(self):
        """
        Devuelve la lista de tipos de canal.
        """
        return self._ch_types

    @property
    def bads(self):
        """
        Devuelve la lista de canales marcados como malos.
        """
        return self._bads

    @property
    def n_channels(self):
        """
        Devuelve la cantidad de canales.
        Se calcula a partir de la longitud de ch_names.
        """
        return len(self._ch_names)

    @property
    def n_samples(self):
        """
        Devuelve la cantidad total de muestras.
        Este valor normalmente lo completa RawSignal cuando carga el archivo.
        """
        return self._n_samples

    @n_samples.setter
    def n_samples(self, value):
        """
        Permite asignar la cantidad de muestras validando que sea un entero
        no negativo o None.
        """
        if value is not None:
            if not isinstance(value, int) or value < 0:
                raise ValueError("n_samples debe ser un entero no negativo.")
        self._n_samples = value

    @property
    def duration(self):
        """
        Calcula la duracion total de la señal en segundos.

        Formula:
            duracion = n_samples / sfreq

        Si todavia no se conoce n_samples, devuelve None.
        """
        if self._n_samples is None:
            return None
        return self._n_samples / self._sfreq

    def __contains__(self, key):
        """
        Permite usar la sintaxis:
            'sfreq' in info

        Devuelve True si la clave existe en el objeto.
        """
        return key in self.keys()

    def __getitem__(self, key):
        """
        Permite acceder como si Info fuera un diccionario:
            info["sfreq"]
            info["ch_names"]

        Si la clave no existe, lanza KeyError.
        """
        data = dict(self.items())
        if key not in data:
            raise KeyError(f"La clave '{key}' no existe en Info.")
        return data[key]

    def __len__(self):
        """
        Devuelve la cantidad de campos de metadata disponibles.
        """
        return len(self.keys())

    def __str__(self):
        """
        Devuelve una representacion legible del objeto cuando se usa print(info).
        Esta salida puede ser util para depuracion o inspeccion rapida.
        """
        return (
            f"Info(ch_names={self.ch_names}, "
            f"n_channels={self.n_channels}, "
            f"sfreq={self.sfreq}, "
            f"n_samples={self.n_samples}, "
            f"duration={self.duration})"
        )

    def get(self, key, default=None):
        """
        Devuelve el valor de una clave si existe.
        Si no existe, devuelve default.

        Ejemplo:
            info.get("sfreq")
            info.get("unknown", "no existe")
        """
        data = dict(self.items())
        return data.get(key, default)

    def keys(self):
        """
        Devuelve la lista de claves disponibles en la metadata.
        Esto permite tratar a Info de forma similar a un diccionario.
        """
        return [
            "ch_names",
            "ch_types",
            "bads",
            "sfreq",
            "n_channels",
            "n_samples",
            "duration",
            "description",
            "experimenter",
            "subject_info",
            "line_freq",
            "lowpass",
            "highpass",
            "notch_freqs",
        ]

    def items(self):
        """
        Devuelve los pares clave-valor de toda la metadata.
        Esto es util para inspeccion, impresion o exportacion.
        """
        return [
            ("ch_names", self.ch_names),
            ("ch_types", self.ch_types),
            ("bads", self.bads),
            ("sfreq", self.sfreq),
            ("n_channels", self.n_channels),
            ("n_samples", self.n_samples),
            ("duration", self.duration),
            ("description", self.description),
            ("experimenter", self.experimenter),
            ("subject_info", self.subject_info),
            ("line_freq", self.line_freq),
            ("lowpass", self.lowpass),
            ("highpass", self.highpass),
            ("notch_freqs", self.notch_freqs),
        ]

    def rename_channels(self, mapping):
        """
        Renombra canales de forma segura usando un diccionario.

        Ejemplo:
            info.rename_channels({"C3": "EEG_C3", "C4": "EEG_C4"})

        Primero verifica que los canales viejos existan.
        Luego reemplaza solo esos nombres.
        Tambien valida que no queden nombres duplicados.
        """
        if not isinstance(mapping, dict):
            raise TypeError("mapping debe ser un diccionario.")

        # Copiamos la lista actual para trabajar sin modificarla directamente
        # hasta haber validado todo correctamente.
        nuevos = self._ch_names.copy()

        # Recorremos cada renombrado solicitado.
        for old_name, new_name in mapping.items():
            if old_name not in nuevos:
                raise ValueError(f"El canal '{old_name}' no esta en la lista de canales.")

            idx = nuevos.index(old_name)
            nuevos[idx] = new_name

        # Verificamos que no queden canales repetidos.
        if len(set(nuevos)) != len(nuevos):
            raise ValueError("Los nombres de canales resultantes no deben repetirse.")

        # Si todo salio bien, actualizamos la lista.
        self._ch_names = nuevos

    def add_bad(self, ch_name):
        """
        Marca un canal como malo.

        Solo se puede marcar como malo un canal que exista.
        No se agregan duplicados.
        """
        if ch_name not in self._ch_names:
            raise ValueError(f"El canal '{ch_name}' no existe.")

        if ch_name not in self._bads:
            self._bads.append(ch_name)

    def remove_bad(self, ch_name):
        """
        Elimina un canal de la lista de canales malos.
        Si no estaba marcado, no hace nada.
        """
        if ch_name in self._bads:
            self._bads.remove(ch_name)