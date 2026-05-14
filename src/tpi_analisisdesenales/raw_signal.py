

#intento de nuevo CAPAZ CONVIENE ACTUALIZAR EL *********************************UML***************

import numpy as np
import pandas as pd

from .info import Info
from .eventos import Eventos
from .anotaciones import Anotaciones


class RawSignal:
    """
    Clase base para representar una señal fisiologica cruda.

    La señal se almacena como una matriz de numpy con forma:
    (n_canales, n_muestras)

    Esta clase guarda:
    - los datos crudos
    - la metadata en un objeto Info
    - los eventos discretos
    - las anotaciones temporales
    """

    def __init__(
        self,
        data: np.ndarray,
        info: Info,
        eventos=None,
        anotaciones=None,
        first_samp: int = 0,
    ):
        """
        Inicializa una instancia de RawSignal.

        Parameters
        ----------
        data : np.ndarray
            Matriz con forma (n_canales, n_muestras).
        info : Info
            Objeto con metadata de la señal.
        eventos : Eventos | None
            Eventos discretos asociados a la señal.
        anotaciones : Anotaciones | None
            Anotaciones temporales asociadas a la señal.
        first_samp : int
            Indice de la primera muestra valida.
        """

        # Convertimos los datos a numpy array para asegurar que el trabajo
        # interno se haga siempre con matrices y no con listas.
        self.data = np.asarray(data, dtype=float)

        # Validamos que la señal tenga dos dimensiones:
        # filas = canales, columnas = muestras
        if self.data.ndim != 2:
            raise ValueError("data debe tener forma (n_canales, n_muestras).")

        # Validamos que info sea una instancia de la clase Info
        if not isinstance(info, Info):
            raise TypeError("info debe ser una instancia de Info.")

        # Verificamos que el numero de canales en data coincida con la metadata
        if self.data.shape[0] != info.n_channels:
            raise ValueError(
                "La cantidad de canales en data no coincide con info.ch_names."
            )

        # Validamos el indice inicial
        if not isinstance(first_samp, int) or first_samp < 0:
            raise ValueError("first_samp debe ser un entero no negativo.")

        # Guardamos metadata y estructuras temporales asociadas
        self.info = info
        self.eventos = eventos if eventos is not None else Eventos()
        self.anotaciones = anotaciones
        self.first_samp = first_samp

        # Guardamos en Info la cantidad de muestras para mantener consistente
        # la metadata de la señal
        self.info.n_samples = self.data.shape[1]

    @property
    def sfreq(self):
        """
        Devuelve la frecuencia de muestreo desde la metadata.
        """
        return self.info.sfreq

    @property
    def duration(self):
        """
        Devuelve la duracion total de la señal en segundos.
        """
        return self.data.shape[1] / self.sfreq

    def _resolve_picks(self, picks=None):
        """
        Convierte picks a una lista de indices de canal.

        Acepta:
        - None -> todos los canales
        - str -> un nombre de canal
        - int -> un indice de canal
        - list[str] o list[int] -> varios canales
        """

        # Si no se especifican canales, se seleccionan todos
        if picks is None:
            return list(range(self.data.shape[0]))

        # Si se pasa un nombre de canal
        if isinstance(picks, str):
            if picks not in self.info.ch_names:
                raise ValueError(f"El canal '{picks}' no existe.")
            return [self.info.ch_names.index(picks)]

        # Si se pasa un indice entero
        if isinstance(picks, int):
            if picks < 0 or picks >= self.data.shape[0]:
                raise IndexError("Indice de canal fuera de rango.")
            return [picks]

        # Si se pasa una lista, resolvemos cada elemento
        if isinstance(picks, list):
            indices = []

            for item in picks:
                if isinstance(item, str):
                    if item not in self.info.ch_names:
                        raise ValueError(f"El canal '{item}' no existe.")
                    indices.append(self.info.ch_names.index(item))

                elif isinstance(item, int):
                    if item < 0 or item >= self.data.shape[0]:
                        raise IndexError("Indice de canal fuera de rango.")
                    indices.append(item)

                else:
                    raise TypeError("Los elementos de picks deben ser str o int.")

            return indices

        raise TypeError("picks debe ser None, str, int o una lista.")

    def get_data(
        self,
        picks=None,
        start: int = 0,
        stop: int | None = None,
        times: bool = False,
        reject: float | None = None,
    ):
        """
        Devuelve datos de la señal con seleccion de canales y recorte temporal.

        Parameters
        ----------
        picks : None, str, int, list[str], list[int]
            Canales a seleccionar.
        start : int
            Muestra inicial.
        stop : int | None
            Muestra final.
        times : bool
            Si es True, devuelve tambien el vector temporal.
        reject : float | None
            Si se indica, elimina canales cuyo pico a pico supere ese valor.

        Returns
        -------
        np.ndarray
            Datos seleccionados.
        tuple[np.ndarray, np.ndarray]
            Si times=True, devuelve datos y tiempos.
        """

        # Si no se pasa stop, tomamos hasta el final
        if stop is None:
            stop = self.data.shape[1]

        # Validamos rango
        if not isinstance(start, int) or not isinstance(stop, int):
            raise TypeError("start y stop deben ser enteros.")
        if start < 0 or stop <= start or stop > self.data.shape[1]:
            raise ValueError("Rango start/stop invalido.")

        # Resolvemos indices de canales
        ch_idx = self._resolve_picks(picks)

        # Extraemos el recorte de datos solicitado
        out = self.data[ch_idx, start:stop]

        # Si se pide reject, descartamos canales con amplitud pico a pico alta
        if reject is not None:
            ptp_values = np.ptp(out, axis=1)
            keep_mask = ptp_values <= reject
            out = out[keep_mask]

        # Si no se piden tiempos, devolvemos solo los datos
        if not times:
            return out

        # Construimos el vector temporal correspondiente al segmento
        t = np.arange(start, stop) / self.sfreq
        return out, t

    def describe(self):
        """
        Devuelve estadisticas descriptivas por canal en un DataFrame.

        Columnas:
        - name
        - type
        - min
        - Q1
        - mediana
        - Q3
        - max
        - rms
        - var
        """

        rows = []

        # Recorremos canal por canal para calcular estadisticas
        for i, ch_name in enumerate(self.info.ch_names):
            channel_data = self.data[i]

            rows.append(
                {
                    "name": ch_name,
                    "type": self.info.ch_types[i],
                    "min": np.min(channel_data),
                    "Q1": np.percentile(channel_data, 25),
                    "mediana": np.median(channel_data),
                    "Q3": np.percentile(channel_data, 75),
                    "max": np.max(channel_data),
                    "rms": np.sqrt(np.mean(channel_data ** 2)),
                    "var": np.var(channel_data),
                }
            )

        return pd.DataFrame(rows)

    def resumen(self):
        """
        Devuelve un resumen general de la señal en forma de diccionario.
        """

        return {
            "n_channels": self.data.shape[0],
            "n_samples": self.data.shape[1],
            "Frecuencia de muestreo": self.sfreq,
            "Duracion": self.duration,
            "Referencia": self.info.get("reference", None),
            "Filtros aplicados": {
                "lowpass": self.info.lowpass,
                "highpass": self.info.highpass,
                "notch_freqs": self.info.notch_freqs,
            },
        }

    def drop_channels(self, ch_names):
        """
        Elimina uno o mas canales de la señal y actualiza la metadata.
        Modifica la instancia actual y retorna self.
        """

        # Validamos entrada
        if not isinstance(ch_names, list) or len(ch_names) == 0:
            raise ValueError("ch_names debe ser una lista no vacia.")

        # Obtenemos los indices de los canales a eliminar
        indices_to_drop = self._resolve_picks(ch_names)

        # Definimos los canales que van a quedar
        indices_to_keep = [
            i for i in range(self.data.shape[0]) if i not in indices_to_drop
        ]

        if len(indices_to_keep) == self.data.shape[0]:
            raise ValueError("No se encontro ningun canal para eliminar.")

        # Filtramos los datos
        self.data = self.data[indices_to_keep, :]

        # Actualizamos metadata asociada a canales
        self.info._ch_names = [self.info.ch_names[i] for i in indices_to_keep]
        self.info._ch_types = [self.info.ch_types[i] for i in indices_to_keep]
        self.info._bads = [ch for ch in self.info.bads if ch in self.info.ch_names]
        self.info.n_samples = self.data.shape[1]

        return self

    def crop(self, tmin: float = 0.0, tmax: float | None = None):
        """
        Recorta la señal en tiempo y modifica la instancia actual.

        Parameters
        ----------
        tmin : float
            Tiempo inicial en segundos.
        tmax : float | None
            Tiempo final en segundos.
        """

        # Si no se pasa tmax, tomamos hasta el final
        if tmax is None:
            tmax = self.duration

        # Validamos intervalo temporal
        if tmin < 0 or tmax <= tmin or tmax > self.duration:
            raise ValueError("Intervalo temporal invalido.")

        # Convertimos tiempo a indices de muestra
        start = int(np.floor(tmin * self.sfreq))
        stop = int(np.ceil(tmax * self.sfreq))

        # Recortamos la señal
        self.data = self.data[:, start:stop]

        # Actualizamos indice inicial y metadata
        self.first_samp += start
        self.info.n_samples = self.data.shape[1]

        # Si hay anotaciones, dejamos solo las que caen en el nuevo rango
        if self.anotaciones is not None and len(self.anotaciones) > 0:
            df = self.anotaciones.get_annotations()
            mask = (df["onset"] >= tmin) & (df["onset"] <= tmax)
            df = df[mask].copy()

            # Reajustamos onsets para que el nuevo recorte empiece en 0
            df["onset"] = df["onset"] - tmin

            self.anotaciones = Anotaciones(
                onset=df["onset"].tolist(),
                duration=df["duration"].tolist(),
                description=df["description"].tolist(),
                ch_names=df["ch_names"].tolist() if "ch_names" in df.columns else None,
            )

        return self
    
    def add_annotation(self, onset, duration, description, ch_names=None):
        if self.anotaciones is None:
            self.anotaciones = Anotaciones()

        self.anotaciones.add(
            onset=onset,
            duration=duration,
            description=description,
            ch_names=ch_names
        )

    def get_channel(self, ch):
        """
        Devuelve un canal individual como array numpy.
        """

        idx = self._resolve_picks(ch)[0]
        return self.data[idx]

    def pick_channels(self, picks):
        """
        Selecciona solo ciertos canales y modifica la instancia actual.
        """

        indices = self._resolve_picks(picks)

        self.data = self.data[indices, :]
        self.info._ch_names = [self.info.ch_names[i] for i in indices]
        self.info._ch_types = [self.info.ch_types[i] for i in indices]
        self.info._bads = [ch for ch in self.info.bads if ch in self.info.ch_names]
        self.info.n_samples = self.data.shape[1]

        return self

    def pick_types(self, **kwargs):
        """
        Selecciona canales por tipo.

        Ejemplo:
            raw.pick_types(eeg=True, ecg=False)
        """

        # Tomamos solo los tipos que el usuario marco como True
        wanted_types = [key for key, value in kwargs.items() if value]

        if len(wanted_types) == 0:
            raise ValueError("Debe indicar al menos un tipo de canal en True.")

        # Buscamos que canales coinciden con esos tipos
        indices = [
            i for i, ch_type in enumerate(self.info.ch_types)
            if ch_type in wanted_types
        ]

        if len(indices) == 0:
            raise ValueError("No hay canales que coincidan con los tipos pedidos.")

        # Filtramos datos y metadata
        self.data = self.data[indices, :]
        self.info._ch_names = [self.info.ch_names[i] for i in indices]
        self.info._ch_types = [self.info.ch_types[i] for i in indices]
        self.info._bads = [ch for ch in self.info.bads if ch in self.info.ch_names]
        self.info.n_samples = self.data.shape[1]

        return self

    def set_anotaciones(self, anotaciones):
        """
        Asocia un objeto Anotaciones a la señal.
        """

        if not isinstance(anotaciones, Anotaciones):
            raise TypeError("anotaciones debe ser una instancia de Anotaciones.")

        self.anotaciones = anotaciones
        return self

    def __getitem__(self, item):
        """
        Permite acceder a la señal como si fuera un array enriquecido.

        Ejemplos:
            raw["C3"]
            raw["Cz", 0:512]
            raw[["C3", "C4"], :]
        """

        # Caso 1: acceso por un solo nombre de canal
        if isinstance(item, str):
            idx = self._resolve_picks(item)
            out = self.data[idx, :]
            t = np.arange(self.data.shape[1]) / self.sfreq
            return out, t

        # Caso 2: acceso con tupla (canal o canales, slice temporal)
        if isinstance(item, tuple) and len(item) == 2:
            picks, time_slice = item

            if not isinstance(time_slice, slice):
                raise TypeError("La segunda parte del indice debe ser un slice.")

            start = 0 if time_slice.start is None else time_slice.start
            stop = self.data.shape[1] if time_slice.stop is None else time_slice.stop

            out = self.get_data(picks=picks, start=start, stop=stop, times=False)
            t = np.arange(start, stop) / self.sfreq
            return out, t

        raise TypeError("Indice no soportado para RawSignal.")

    def __str__(self):
        """
        Representacion textual de la señal.
        """

        r = self.resumen()
        return (
            f"RawSignal("
            f"n_channels={r['n_channels']}, "
            f"n_samples={r['n_samples']}, "
            f"sfreq={r['Frecuencia de muestreo']}, "
            f"duration={r['Duracion']:.3f} s)"
        )