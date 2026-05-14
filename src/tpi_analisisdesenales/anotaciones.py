import json
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd


class Anotaciones:
    """
    Clase para almacenar y gestionar anotaciones en registros fisiologicos.

    Cada anotacion representa un intervalo temporal definido por:
    - onset: instante de inicio en segundos
    - duration: duracion en segundos
    - description: descripcion textual del evento o situacion observada

    Tambien puede asociarse opcionalmente una lista de canales implicados
    en cada anotacion.
    """

    def __init__(
        self,
        onset: Optional[npt.NDArray[np.floating] | Sequence[float]] = None,
        duration: Optional[Sequence[float]] = None,
        description: Optional[Sequence[str]] = None,
        t0: Optional[float] = None,
        ch_names: Optional[Sequence[Sequence[str]]] = None,
    ) -> None:
        """
        Inicializa una instancia de la clase Anotaciones.

        Parameters
        ----------
        onset : array o secuencia de float
            Tiempos de inicio de las anotaciones en segundos.
        duration : secuencia de float
            Duracion de cada anotacion en segundos.
        description : secuencia de str
            Descripcion de cada anotacion.
        t0 : float, optional
            Tiempo inicial del registro. Se guarda como referencia.
        ch_names : secuencia de secuencias, optional
            Canales asociados a cada anotacion.
            Ejemplo: [[], ["C3"], ["C4", "Cz"]]
        """
        if onset is None and duration is None and description is None:
            self._onset = np.array([], dtype=float)
            self._duration = []
            self._description = []
            self._t0 = t0
            self._ch_names = []
            return

        if onset is None or duration is None or description is None:
            raise ValueError("onset, duration y description deben ingresarse juntos.")

        # Convertimos onset a numpy array para poder validarlo y manipularlo
        # de forma uniforme aunque el usuario pase una lista.
        onset_array = np.asarray(onset, dtype=float)

        # Convertimos duration a lista de float.
        duration_list = [float(value) for value in duration]

        # Convertimos description a lista de str.
        description_list = [str(value) for value in description]

        # Si no se pasan canales asociados, generamos una lista vacia
        # para cada anotacion. Esto evita tener un atributo inconsistente.
        if ch_names is None:
            ch_names_list = [[] for _ in range(len(onset_array))]
        else:
            ch_names_list = [list(ch_list) for ch_list in ch_names]

        # Validamos que todas las estructuras tengan la misma longitud.
        # Cada onset debe corresponder con una duration, una description
        # y una lista de canales asociados.
        n_annotations = len(onset_array)
        if len(duration_list) != n_annotations:
            raise ValueError("duration debe tener la misma longitud que onset.")
        if len(description_list) != n_annotations:
            raise ValueError("description debe tener la misma longitud que onset.")
        if len(ch_names_list) != n_annotations:
            raise ValueError("ch_names debe tener la misma longitud que onset.")

        # Validamos que los onset no sean negativos.
        if np.any(onset_array < 0):
            raise ValueError("Los valores de onset no pueden ser negativos.")

        # Validamos que las duraciones no sean negativas.
        if any(value < 0 for value in duration_list):
            raise ValueError("Los valores de duration no pueden ser negativos.")

        # Validamos que cada descripcion no este vacia.
        if any(desc.strip() == "" for desc in description_list):
            raise ValueError("Ninguna description puede estar vacia.")

        # Guardamos los datos internamente.
        # Se usan atributos protegidos para mantener cierto control
        # y facilitar futuras extensiones con propiedades o validaciones.
        self._onset = onset_array
        self._duration = duration_list
        self._description = description_list
        self._t0 = t0
        self._ch_names = ch_names_list

    def __len__(self) -> int:
        """
        Devuelve la cantidad total de anotaciones.
        """
        return len(self._onset)

    def __iter__(self) -> Iterator[dict]:
        """
        Permite iterar sobre las anotaciones una por una.

        Cada iteracion devuelve un diccionario con los datos
        de una anotacion individual.
        """
        for i in range(len(self)):
            yield {
                "onset": self._onset[i],
                "duration": self._duration[i],
                "description": self._description[i],
                "ch_names": self._ch_names[i],
            }

    def __str__(self) -> str:
        """
        Devuelve una representacion legible del objeto.

        En lugar de imprimir toda la tabla completa, resume cuantas
        anotaciones hay y cuantas veces aparece cada descripcion.
        """
        if len(self) == 0:
            return "Anotaciones(0 anotaciones)"

        # Creamos un DataFrame temporal para contar ocurrencias por descripcion.
        df = self.get_annotations()
        counts = df["description"].value_counts()

        # Armamos un resumen textual.
        summary_parts = [f"{desc}: {count}" for desc, count in counts.items()]
        summary_text = ", ".join(summary_parts)

        return f"Anotaciones({len(self)} anotaciones | {summary_text})"

    @property
    def onset(self) -> npt.NDArray[np.float64]:
        """
        Devuelve los tiempos de inicio como array de numpy.
        """
        return self._onset

    @property
    def duration(self) -> list[float]:
        """
        Devuelve la lista de duraciones.
        """
        return self._duration

    @property
    def description(self) -> list[str]:
        """
        Devuelve la lista de descripciones.
        """
        return self._description

    @property
    def ch_names(self) -> list[list[str]]:
        """
        Devuelve la lista de canales asociados a cada anotacion.
        """
        return self._ch_names

    @property
    def t0(self) -> Optional[float]:
        """
        Devuelve el tiempo inicial del registro si existe.
        """
        return self._t0

    def add(
        self,
        onset: float,
        duration: float,
        description: str,
        ch_names: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Agrega una nueva anotacion al objeto.

        Este metodo modifica la instancia actual.
        """

        if onset < 0:
            raise ValueError("onset no puede ser negativo.")

        if duration < 0:
            raise ValueError("duration no puede ser negativo.")

        if not isinstance(description, str) or description.strip() == "":
            raise ValueError("description debe ser un texto no vacio.")

        if ch_names is None:
            ch_names_list = []
        else:
            if isinstance(ch_names, str):
                raise TypeError("ch_names debe ser una secuencia de nombres, no un string.")

            ch_names_list = list(ch_names)

            if len(ch_names_list) == 0:
                raise ValueError("ch_names no puede estar vacio si se especifica.")

            for ch in ch_names_list:
                if not isinstance(ch, str) or ch.strip() == "":
                    raise ValueError("Cada canal en ch_names debe ser un string no vacio.")

        self._onset = np.append(self._onset, float(onset))
        self._duration.append(float(duration))
        self._description.append(description)
        self._ch_names.append(ch_names_list)

    def append(
        self,
        onset: float,
        duration: float,
        description: str,
        ch_names: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Alias de add().

        Se deja porque la consigna menciona explicitamente append().
        """
        self.add(
            onset=onset,
            duration=duration,
            description=description,
            ch_names=ch_names,
        )

    def remove(
        self,
        index: Optional[int] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Elimina una o varias anotaciones.

        Puede eliminar:
        - por indice
        - por descripcion

        Debe pasarse exactamente uno de los dos parametros.
        """
        # Verificamos que el usuario no intente mezclar criterios.
        if (index is None and description is None) or (
            index is not None and description is not None
        ):
            raise ValueError("Debe indicar solo index o solo description.")

        # Caso 1: eliminar por indice.
        if index is not None:
            if not isinstance(index, int):
                raise TypeError("index debe ser un entero.")
            if index < 0 or index >= len(self):
                raise IndexError("index fuera de rango.")

            # Eliminamos en todas las estructuras para mantener coherencia.
            self._onset = np.delete(self._onset, index)
            del self._duration[index]
            del self._description[index]
            del self._ch_names[index]
            return

        # Caso 2: eliminar por descripcion.
        # Se eliminan todas las anotaciones que coincidan exactamente.
        indices_to_keep = [
            i for i, desc in enumerate(self._description) if desc != description
        ]

        # Si no cambia nada, es porque no existia esa descripcion.
        if len(indices_to_keep) == len(self):
            raise ValueError(f"No existe ninguna anotacion con description='{description}'.")

        # Reconstruimos las estructuras filtrando solo las anotaciones
        # que deben mantenerse.
        self._onset = self._onset[indices_to_keep]
        self._duration = [self._duration[i] for i in indices_to_keep]
        self._description = [self._description[i] for i in indices_to_keep]
        self._ch_names = [self._ch_names[i] for i in indices_to_keep]

    def get_annotations(self) -> pd.DataFrame:
        """
        Devuelve todas las anotaciones como un DataFrame.

        Esto facilita busquedas, impresion y exportacion.
        """
        return pd.DataFrame(
            {
                "onset": self._onset,
                "duration": self._duration,
                "description": self._description,
                "ch_names": self._ch_names,
            }
        )

    def find(self, value: str | int | float) -> pd.DataFrame:
        """
        Busca anotaciones por descripcion, onset o duration.

        Parameters
        ----------
        value : str | int | float
            Valor a buscar.

        Returns
        -------
        pd.DataFrame
            DataFrame con las anotaciones encontradas.
        """
        df = self.get_annotations()

        # Si el valor es texto, buscamos por descripcion exacta.
        if isinstance(value, str):
            result = df[df["description"] == value]
            return result.reset_index(drop=True)

        # Si el valor es numerico, buscamos coincidencia en onset o duration.
        if isinstance(value, (int, float)):
            result = df[(df["onset"] == float(value)) | (df["duration"] == float(value))]
            return result.reset_index(drop=True)

        # Si el valor no es de un tipo soportado, lanzamos error.
        raise TypeError("value debe ser str, int o float.")

    def save(self, filepath: str | Path) -> None:
        """
        Guarda las anotaciones en un archivo.

        Formatos soportados:
        - .csv
        - .json
        - .txt

        En .txt se guarda como tabla separada por tabulaciones.
        """
        # Convertimos la ruta a objeto Path para trabajar mejor con ella.
        filepath = Path(filepath)

        # Obtenemos el DataFrame con las anotaciones.
        df = self.get_annotations()

        # Detectamos la extension y guardamos segun corresponda.
        if filepath.suffix.lower() == ".csv":
            df.to_csv(filepath, index=False)
            return

        if filepath.suffix.lower() == ".json":
            # Para json usamos orient="records" para obtener una lista
            # de diccionarios, facil de reutilizar luego.
            df.to_json(filepath, orient="records", force_ascii=False, indent=4)
            return

        if filepath.suffix.lower() == ".txt":
            df.to_csv(filepath, index=False, sep="\t")
            return

        # Si la extension no esta soportada, se informa claramente.
        raise ValueError("Formato no soportado. Use .csv, .json o .txt")

    @classmethod
    def load(cls, filepath: str | Path) -> "Anotaciones":
        """
        Carga anotaciones desde un archivo y devuelve una nueva instancia.

        Formatos soportados:
        - .csv
        - .json
        - .txt
        """
        # Convertimos a Path para simplificar el manejo de la ruta.
        filepath = Path(filepath)

        # Verificamos que el archivo exista antes de intentar leerlo.
        if not filepath.exists():
            raise FileNotFoundError(f"No existe el archivo: {filepath}")

        # Leemos segun la extension detectada.
        if filepath.suffix.lower() == ".csv":
            df = pd.read_csv(filepath)

        elif filepath.suffix.lower() == ".json":
            df = pd.read_json(filepath)

        elif filepath.suffix.lower() == ".txt":
            df = pd.read_csv(filepath, sep="\t")

        else:
            raise ValueError("Formato no soportado. Use .csv, .json o .txt")

        # Validamos que al menos existan las columnas basicas.
        required_columns = {"onset", "duration", "description"}
        if not required_columns.issubset(df.columns):
            raise ValueError(
                "El archivo no contiene las columnas requeridas: "
                "onset, duration, description."
            )

        # Si no viene la columna ch_names, la generamos vacia.
        # Esto deja la clase flexible para archivos mas simples.
        if "ch_names" not in df.columns:
            df["ch_names"] = [[] for _ in range(len(df))]
        else:
            # Si viene desde csv o txt, ch_names puede quedar como string.
            # Intentamos convertirla de forma razonable.
            parsed_ch_names = []
            for value in df["ch_names"]:
                if isinstance(value, list):
                    parsed_ch_names.append(value)
                elif pd.isna(value):
                    parsed_ch_names.append([])
                elif isinstance(value, str):
                    # Intentamos interpretar listas serializadas.
                    try:
                        parsed_value = json.loads(value)
                        if isinstance(parsed_value, list):
                            parsed_ch_names.append(parsed_value)
                        else:
                            parsed_ch_names.append([value])
                    except json.JSONDecodeError:
                        parsed_ch_names.append([value])
                else:
                    parsed_ch_names.append([str(value)])
            df["ch_names"] = parsed_ch_names

        # Construimos la nueva instancia usando los datos cargados.
        return cls(
            onset=df["onset"].to_numpy(dtype=float),
            duration=df["duration"].tolist(),
            description=df["description"].tolist(),
            ch_names=df["ch_names"].tolist(),
        )