import numpy as np


class Epocas:
    """
    Representa segmentos de senal extraidos alrededor de eventos.

    Cada epoca se obtiene a partir de un evento y una ventana temporal
    definida por tmin y tmax.
    """

    def __init__(
        self,
        raw,
        eventos,
        event_id=None,
        tmin: float = -0.2,
        tmax: float = 0.8,
    ) -> None:
        """
        Crea epocas a partir de una senal continua y sus eventos.

        Parameters
        ----------
        raw
            Objeto RawSignal que contiene la senal continua.

        eventos
            Objeto Eventos con las marcas de eventos.

        event_id : int | list[int] | None
            Identificador o lista de identificadores de eventos a utilizar.
            Si es None, se usan todos los eventos.

        tmin : float
            Tiempo inicial de la epoca respecto al evento, en segundos.

        tmax : float
            Tiempo final de la epoca respecto al evento, en segundos.
        """

        if tmax <= tmin:
            raise ValueError("tmax debe ser mayor que tmin.")

        self.raw = raw
        self.eventos = eventos
        self.event_id = event_id
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.sfreq = raw.sfreq
        self.info = raw.info

        self.data, self.selected_events = self._create_epochs()

    def _create_epochs(self):
        """
        Extrae las epocas desde la senal continua.
        """

        eventos_df = self.eventos.get_events()

        if self.event_id is not None:
            if isinstance(self.event_id, int):
                event_ids = [self.event_id]
            else:
                event_ids = list(self.event_id)

            eventos_df = eventos_df[eventos_df["event_id"].isin(event_ids)]

        n_pre = int(round(self.tmin * self.sfreq))
        n_post = int(round(self.tmax * self.sfreq))
        n_times = n_post - n_pre

        epocas = []
        eventos_validos = []

        for _, evento in eventos_df.iterrows():
            sample = int(evento["sample"])

            start = sample + n_pre - self.raw.first_samp
            stop = sample + n_post - self.raw.first_samp

            if start < 0 or stop > self.raw.data.shape[1]:
                continue

            epoca = self.raw.data[:, start:stop]

            if epoca.shape[1] == n_times:
                epocas.append(epoca)
                eventos_validos.append(evento)

        if len(epocas) == 0:
            data = np.empty((0, self.raw.data.shape[0], n_times))
        else:
            data = np.stack(epocas, axis=0)

        return data, eventos_validos

    def get_data(self):
        """
        Devuelve los datos de las epocas.

        Shape:
        n_epocas x n_canales x n_muestras
        """

        return self.data

    def average(self):
        """
        Calcula el promedio de todas las epocas.

        Devuelve un arreglo de shape:
        n_canales x n_muestras
        """

        if len(self) == 0:
            raise ValueError("No hay epocas para promediar.")

        return np.mean(self.data, axis=0)

    def __len__(self):
        """
        Devuelve la cantidad de epocas extraidas.
        """

        return self.data.shape[0]

    def __repr__(self):
        return (
            f"Epocas(n_epocas={len(self)}, "
            f"n_canales={self.data.shape[1]}, "
            f"n_muestras={self.data.shape[2]}, "
            f"tmin={self.tmin}, tmax={self.tmax})"
        )
    
