
import numpy as np
import pandas as pd

from .raw_signal import RawSignal
from .anotaciones import Anotaciones
from .eventos import Eventos
from .info import Info


class EMGSignal(RawSignal):
    """
    Clase para representar y analizar señales EMG.

    Hereda la funcionalidad general de RawSignal y agrega
    metodos especificos para analisis basico de electromiografia.
    """

    def __init__(
        self,
        data: np.ndarray,
        info: Info,
        eventos: Eventos | None = None,
        anotaciones: Anotaciones | None = None,
        first_samp: int = 0,
        muscle_group: str | None = None,
        sampling_window: int | None = None,
        filters_applied: list | None = None,
        features: dict | None = None,
    ):
        """"
        Inicializa una instancia de EMGSignal.

        Parameters
        ----------
        data : np.ndarray
            Matriz con forma (n_canales, n_muestras).
        info : Info
            Objeto con metadata de la señal.
        eventos : Eventos | None
            Eventos discretos asociados.
        anotaciones : Anotaciones | None
            Anotaciones temporales asociadas.
        first_samp : int
            Indice de la primera muestra valida.
        muscle_group : str | None
            Nombre general del grupo muscular registrado.
        sampling_window : int | None
            Tamaño de ventana sugerido para procesamiento.
        filters_applied : list | None
            Lista con filtros aplicados a la señal.
        features : dict | None
            Diccionario para guardar caracteristicas extraidas.
        """

        # Inicializamos la parte heredada de RawSignal
        super().__init__(
            data=data,
            info=info,
            eventos=eventos,
            anotaciones=anotaciones,
            first_samp=first_samp,
        )

        # Atributos especificos de EMG
        self.muscle_group = muscle_group
        self.sampling_window = sampling_window
        self.filters_applied = filters_applied if filters_applied is not None else []
        self.features = features if features is not None else {}

        # Validamos que la estructura de la señal sea coherente
        self.validate_structure()

    def validate_structure(self):
        """
        Valida la estructura basica de la señal EMG.

        Verifica:
        - que data sea 2D
        - que haya al menos un canal y una muestra
        - que los tipos de canal sean coherentes con EMG
        """
        if self.data.ndim != 2:
            raise ValueError("La señal EMG debe ser una matriz 2D.")

        if self.data.shape[0] < 1:
            raise ValueError("La señal EMG debe tener al menos un canal.")

        if self.data.shape[1] < 1:
            raise ValueError("La señal EMG debe tener al menos una muestra.")

        if len(self.info.ch_names) != self.data.shape[0]:
            raise ValueError(
                "La cantidad de nombres de canal no coincide con la señal."
            )

        if len(self.info.ch_types) != self.data.shape[0]:
            raise ValueError(
                "La cantidad de tipos de canal no coincide con la señal."
            )

        return True

    def rms_per_channel(self):
        """
        Calcula el RMS por canal.

        Returns
        -------
        np.ndarray
            Vector con un valor RMS por canal.
        """
        rms = np.sqrt(np.mean(self.data ** 2, axis=1))
        return rms

    def envelope(self, rectify: bool = True):
        """
        Calcula una envolvente basica de la señal EMG.

        En esta primera version, la envolvente se calcula como
        el valor absoluto de la señal si rectify=True.

        Parameters
        ----------
        rectify : bool
            Si es True, devuelve la señal rectificada.

        Returns
        -------
        np.ndarray
            Señal procesada canal por canal.
        """
        if rectify:
            return np.abs(self.data)

        return self.data.copy()

    def describe(self):
        """
        Devuelve estadisticas descriptivas por canal para EMG.

        Extiende describe() de RawSignal agregando una columna
        especifica con RMS por canal.
        """
        df = super().describe()
        df["emg_rms"] = self.rms_per_channel()
        return df

    def feature_extraction(self):
        """
        Extrae un conjunto basico de caracteristicas EMG y las guarda.

        Por ahora:
        - rms por canal
        - valor medio absoluto por canal
        - varianza por canal
        """
        rms = self.rms_per_channel()
        mav = np.mean(np.abs(self.data), axis=1)
        var = np.var(self.data, axis=1)

        self.features = {
            "rms": rms,
            "mav": mav,
            "var": var,
        }

        return self.features