from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd


class Eventos:
    """
    Clase para almacenar y gestionar eventos discretos dentro de una señal.

    Cada evento queda definido por:
    - sample: muestra en la que ocurre
    - event_id: identificador entero del evento

    Opcionalmente se puede usar event_dict para mapear nombres a ids
    o ids a nombres, por ejemplo:
        {"izquierda": 1, "derecha": 2}
    o
        {1: "izquierda", 2: "derecha"}
    """

    def __init__(
        self,
        samples: Optional[npt.NDArray[np.integer] | list[int]] = None,
        event_id: Optional[npt.NDArray[np.integer] | list[int]] = None,
        event_dict: Optional[dict[str, int] | dict[int, str]] = None,
    ) -> None:
        """
        Inicializa una instancia de la clase Eventos.

        Parameters
        ----------
        samples : array o lista de enteros, optional
            Muestras donde ocurren los eventos.
        event_id : array o lista de enteros, optional
            Identificadores de cada evento.
        event_dict : dict, optional
            Diccionario de mapeo entre nombre e identificador.
        """

        # Si no se pasan samples o event_id, inicializamos estructuras vacias.
        if samples is None:
            samples = []
        if event_id is None:
            event_id = []

        # Convertimos a numpy arrays para trabajar de forma uniforme.
        samples_array = np.asarray(samples, dtype=int)
        event_id_array = np.asarray(event_id, dtype=int)

        # Validamos que ambas estructuras tengan la misma longitud.
        if len(samples_array) != len(event_id_array):
            raise ValueError("samples y event_id deben tener la misma longitud.")

        # Validamos que las muestras no sean negativas.
        if np.any(samples_array < 0):
            raise ValueError("Las muestras no pueden ser negativas.")

        # Guardamos atributos internos.
        self._samples = samples_array
        self._event_id = event_id_array
        self._event_dict = event_dict if event_dict is not None else {}

    @property
    def samples(self) -> npt.NDArray[np.int64]:
        """
        Devuelve el array de muestras de los eventos.
        """
        return self._samples

    @property
    def event_id(self) -> npt.NDArray[np.int64]:
        """
        Devuelve el array de ids de los eventos.
        """
        return self._event_id

    @property
    def event_dict(self) -> dict:
        """
        Devuelve el diccionario de mapeo de eventos.
        """
        return self._event_dict

    def __len__(self) -> int:
        """
        Devuelve la cantidad total de eventos.
        """
        return len(self._samples)

    def __str__(self) -> str:
        """
        Devuelve una representacion legible del objeto.
        """
        if len(self) == 0:
            return "Eventos(0 eventos)"
        return f"Eventos({len(self)} eventos, ids_unicos={sorted(set(self._event_id.tolist()))})"

    def add(self, sample: int, event_id: int) -> None:
        """
        Agrega un nuevo evento.

        Parameters
        ----------
        sample : int
            Muestra donde ocurre el evento.
        event_id : int
            Identificador del evento.
        """
        # Validamos que ambos sean enteros.
        if not isinstance(sample, int):
            raise TypeError("sample debe ser un entero.")
        if not isinstance(event_id, int):
            raise TypeError("event_id debe ser un entero.")

        # Validamos que la muestra no sea negativa.
        if sample < 0:
            raise ValueError("sample no puede ser negativo.")

        # Agregamos el nuevo evento a las estructuras internas.
        self._samples = np.append(self._samples, sample)
        self._event_id = np.append(self._event_id, event_id)

    def remove(
        self,
        index: Optional[int] = None,
        sample: Optional[int] = None,
        event_id: Optional[int] = None,
    ) -> None:
        """
        Elimina eventos segun un criterio.

        Se puede eliminar:
        - por indice
        - por sample
        - por event_id

        Solo debe pasarse uno de esos criterios.
        """
        criterios = [index is not None, sample is not None, event_id is not None]

        if sum(criterios) != 1:
            raise ValueError("Debe indicar exactamente uno entre index, sample o event_id.")

        # Eliminar por indice.
        if index is not None:
            if not isinstance(index, int):
                raise TypeError("index debe ser un entero.")
            if index < 0 or index >= len(self):
                raise IndexError("index fuera de rango.")

            self._samples = np.delete(self._samples, index)
            self._event_id = np.delete(self._event_id, index)
            return

        # Eliminar por muestra.
        if sample is not None:
            mask = self._samples != sample
            if np.all(mask):
                raise ValueError(f"No existe ningun evento en la muestra {sample}.")
            self._samples = self._samples[mask]
            self._event_id = self._event_id[mask]
            return

        # Eliminar por id de evento.
        if event_id is not None:
            mask = self._event_id != event_id
            if np.all(mask):
                raise ValueError(f"No existe ningun evento con event_id={event_id}.")
            self._samples = self._samples[mask]
            self._event_id = self._event_id[mask]

    def find(self, event_id: int) -> npt.NDArray[np.int64]:
        """
        Devuelve las muestras donde ocurren los eventos de un id dado.

        Parameters
        ----------
        event_id : int
            Identificador del evento a buscar.

        Returns
        -------
        np.ndarray
            Array con las muestras encontradas.
        """
        if not isinstance(event_id, int):
            raise TypeError("event_id debe ser un entero.")

        return self._samples[self._event_id == event_id]

    def get_events(self) -> pd.DataFrame:
        """
        Devuelve todos los eventos en un DataFrame.
        """
        return pd.DataFrame(
            {
                "sample": self._samples,
                "event_id": self._event_id,
            }
        )

    def as_array(self) -> npt.NDArray[np.int64]:
        """
        Devuelve los eventos como array con shape (n_eventos, 2).

        Formato:
            [[sample_1, event_id_1],
             [sample_2, event_id_2],
             ...]
        """
        if len(self) == 0:
            return np.empty((0, 2), dtype=int)

        return np.column_stack((self._samples, self._event_id))

    def save(self, filepath: str | Path) -> None:
        """
        Guarda los eventos en un archivo .csv o .txt.
        """
        filepath = Path(filepath)
        df = self.get_events()

        if filepath.suffix.lower() == ".csv":
            df.to_csv(filepath, index=False)
            return

        if filepath.suffix.lower() == ".txt":
            df.to_csv(filepath, index=False, sep="\t")
            return

        raise ValueError("Formato no soportado. Use .csv o .txt")

    @classmethod
    def load(
        cls,
        filepath: str | Path,
        event_dict: Optional[dict[str, int] | dict[int, str]] = None,
    ) -> "Eventos":
        """
        Carga eventos desde un archivo .csv o .txt y devuelve una nueva instancia.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"No existe el archivo: {filepath}")

        if filepath.suffix.lower() == ".csv":
            df = pd.read_csv(filepath)
        elif filepath.suffix.lower() == ".txt":
            df = pd.read_csv(filepath, sep="\t")
        else:
            raise ValueError("Formato no soportado. Use .csv o .txt")

        required_columns = {"sample", "event_id"}
        if not required_columns.issubset(df.columns):
            raise ValueError("El archivo debe contener las columnas 'sample' y 'event_id'.")

        return cls(
            samples=df["sample"].to_numpy(dtype=int),
            event_id=df["event_id"].to_numpy(dtype=int),
            event_dict=event_dict,
        )