import numpy as np
import plotly.graph_objects as go


class PlotEngine:
    """
    Motor grafico reutilizable para señales fisiologicas.

    Esta clase no almacena logica de procesamiento.
    Solo recibe datos ya cargados y genera visualizaciones.
    La idea es que luego RawSignal, EEGSignal, ECGSignal o EMGSignal
    llamen a este motor en lugar de duplicar codigo de graficado.
    """

    def __init__(self, data, sfreq, ch_names, anotaciones=None):
        """
        Inicializa el motor grafico.

        Parameters
        ----------
        data : np.ndarray
            Matriz de datos con forma (n_canales, n_muestras).
        sfreq : float
            Frecuencia de muestreo en Hz.
        ch_names : list[str]
            Lista con nombres de los canales.
        anotaciones : objeto opcional
            Objeto de anotaciones. Debe poder recorrerse y devolver
            diccionarios con onset, duration y description.
        """

        # Guardamos los datos principales del grafico.
        self.data = np.asarray(data, dtype=float)
        self.sfreq = float(sfreq)
        self.ch_names = list(ch_names)
        self.anotaciones = anotaciones

        # Validamos que todo tenga coherencia.
        self._validate_inputs()

        # Construimos el vector temporal una sola vez.
        self.times = self._build_time_vector()

    def _validate_inputs(self):
        """
        Valida que los datos tengan el formato correcto.
        """

        # La señal debe ser una matriz 2D: canales x muestras.
        if self.data.ndim != 2:
            raise ValueError("data debe tener forma (n_canales, n_muestras).")

        # La frecuencia de muestreo debe ser positiva.
        if self.sfreq <= 0:
            raise ValueError("sfreq debe ser mayor que cero.")

        # Debe haber un nombre por cada canal.
        if len(self.ch_names) != self.data.shape[0]:
            raise ValueError(
                "La cantidad de nombres de canal no coincide con la cantidad de canales."
            )

    def _build_time_vector(self):
        """
        Genera el vector temporal en segundos.
        """

        # Si hay N muestras y la frecuencia es sfreq, el tiempo es:
        # 0, 1/sfreq, 2/sfreq, ..., (N-1)/sfreq
        n_samples = self.data.shape[1]
        return np.arange(n_samples) / self.sfreq

    def _resolve_picks(self, picks=None):
        """
        Convierte picks a indices reales de canales.

        Parameters
        ----------
        picks : None, str, int, list[str], list[int]
            Seleccion de canales.

        Returns
        -------
        list[int]
            Lista de indices de canales.
        """

        # Si no se especifica nada, usamos todos los canales.
        if picks is None:
            return list(range(len(self.ch_names)))

        # Si se pasa un solo canal como string.
        if isinstance(picks, str):
            if picks not in self.ch_names:
                raise ValueError(f"El canal '{picks}' no existe.")
            return [self.ch_names.index(picks)]

        # Si se pasa un solo indice entero.
        if isinstance(picks, int):
            if picks < 0 or picks >= len(self.ch_names):
                raise IndexError("Indice de canal fuera de rango.")
            return [picks]

        # Si se pasa una lista, resolvemos cada elemento.
        if isinstance(picks, list):
            indices = []

            for item in picks:
                if isinstance(item, str):
                    if item not in self.ch_names:
                        raise ValueError(f"El canal '{item}' no existe.")
                    indices.append(self.ch_names.index(item))

                elif isinstance(item, int):
                    if item < 0 or item >= len(self.ch_names):
                        raise IndexError("Indice de canal fuera de rango.")
                    indices.append(item)

                else:
                    raise TypeError(
                        "Los elementos de picks deben ser str o int."
                    )

            return indices

        raise TypeError("picks debe ser None, str, int o una lista.")

    def _time_to_sample(self, start=0.0, stop=None):
        """
        Convierte un intervalo temporal a indices de muestra.

        Parameters
        ----------
        start : float
            Tiempo inicial en segundos.
        stop : float | None
            Tiempo final en segundos.

        Returns
        -------
        tuple[int, int]
            Indices de inicio y fin.
        """

        # Si stop no se pasa, tomamos hasta el final de la señal.
        if stop is None:
            stop = self.times[-1]

        # Validamos el rango de tiempo.
        if start < 0:
            raise ValueError("start no puede ser negativo.")
        if stop <= start:
            raise ValueError("stop debe ser mayor que start.")
        if stop > self.times[-1] + (1 / self.sfreq):
            raise ValueError("stop excede la duracion de la señal.")

        # Convertimos segundos a muestras.
        idx_start = int(np.floor(start * self.sfreq))
        idx_stop = int(np.ceil(stop * self.sfreq))

        # Ajustamos para no salirnos del tamaño real.
        idx_start = max(0, idx_start)
        idx_stop = min(self.data.shape[1], idx_stop)

        return idx_start, idx_stop

    def _get_segment(self, picks=None, start=0.0, stop=None):
        """
        Obtiene el segmento de señal y su vector temporal asociado.

        Returns
        -------
        segment_data : np.ndarray
            Datos recortados.
        segment_times : np.ndarray
            Vector temporal del recorte.
        indices : list[int]
            Indices de canales seleccionados.
        """

        # Resolvemos canales.
        indices = self._resolve_picks(picks)

        # Resolvemos intervalo de tiempo.
        idx_start, idx_stop = self._time_to_sample(start, stop)

        # Extraemos solo el tramo visible.
        segment_data = self.data[indices, idx_start:idx_stop]
        segment_times = self.times[idx_start:idx_stop]

        return segment_data, segment_times, indices

    def _get_visible_annotations(self, start=0.0, stop=None):
        """
        Devuelve solo las anotaciones que caen dentro de la ventana visible.
        """

        # Si no hay anotaciones, devolvemos lista vacia.
        if self.anotaciones is None:
            return []

        # Si stop es None, tomamos el final de la señal.
        if stop is None:
            stop = self.times[-1]

        visibles = []

        # Recorremos las anotaciones.
        for anot in self.anotaciones:
            onset = float(anot["onset"])
            duration = float(anot["duration"])
            description = str(anot["description"])

            # Consideramos visible si el inicio cae dentro de la ventana.
            if start <= onset <= stop:
                visibles.append(
                    {
                        "onset": onset,
                        "duration": duration,
                        "description": description,
                    }
                )

        return visibles

    def plot_signals(
        self,
        picks=None,
        start=0.0,
        stop=None,
        superpose=False,
        show_annotations=True,
        fill_annotations=True,
        title="Visualizacion de señal",
    ):
        """
        Grafica uno o varios canales.

        Parameters
        ----------
        picks : None, str, int, list[str], list[int]
            Canales a mostrar.
        start : float
            Tiempo inicial en segundos.
        stop : float | None
            Tiempo final en segundos.
        superpose : bool
            Si es True, superpone los canales.
            Si es False, los separa verticalmente.
        show_annotations : bool
            Si es True, dibuja anotaciones.
        fill_annotations : bool
            Si es True, sombrea la duracion de la anotacion.
        title : str
            Titulo del grafico.

        Returns
        -------
        plotly.graph_objects.Figure
            Figura generada.
        """

        # Obtenemos el recorte a mostrar.
        segment_data, segment_times, indices = self._get_segment(
            picks=picks,
            start=start,
            stop=stop,
        )

        # Creamos la figura.
        fig = go.Figure()

        # Este desplazamiento vertical se usa solo si NO superponemos.
        # Asi evitamos que todas las curvas queden una encima de otra.
        offset_step = 0.0
        if not superpose:
            # Elegimos un desplazamiento segun la amplitud del recorte.
            # Si la señal es casi plana, usamos un valor minimo.
            max_amp = np.max(np.abs(segment_data)) if segment_data.size > 0 else 1.0
            offset_step = max(max_amp * 2.5, 1.0)

        # Agregamos cada canal como una traza.
        for i, ch_idx in enumerate(indices):
            y = segment_data[i].copy()

            # Si no queremos superponer, desplazamos cada canal.
            if not superpose:
                y = y + i * offset_step

            fig.add_trace(
                go.Scatter(
                    x=segment_times,
                    y=y,
                    mode="lines",
                    name=self.ch_names[ch_idx],
                )
            )

        # Agregamos anotaciones si corresponde.
        if show_annotations:
            visibles = self._get_visible_annotations(start=start, stop=stop)

            for anot in visibles:
                onset = anot["onset"]
                duration = anot["duration"]
                description = anot["description"]

                # Linea vertical en el onset.
                fig.add_vline(
                    x=onset,
                    line_dash="dash",
                    annotation_text=description,
                    annotation_position="top right",
                )

                # Sombreado opcional de la duracion.
                if fill_annotations and duration > 0:
                    fig.add_vrect(
                        x0=onset,
                        x1=onset + duration,
                        opacity=0.15,
                        line_width=0,
                    )

        # Configuramos layout general.
        fig.update_layout(
            title=title,
            xaxis_title="Tiempo [s]",
            yaxis_title="Amplitud",
            hovermode="x unified",
        )

        # Agregamos slider para navegar mas comodamente.
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="linear",
            )
        )

        return fig

    def plot_mean_std(
        self,
        picks=None,
        start=0.0,
        stop=None,
        title="Media y desvio estandar",
    ):
        """
        Grafica la media y el desvio estandar del conjunto de canales seleccionados.

        Parameters
        ----------
        picks : None, str, int, list[str], list[int]
            Canales a incluir en el calculo.
        start : float
            Tiempo inicial en segundos.
        stop : float | None
            Tiempo final en segundos.
        title : str
            Titulo del grafico.

        Returns
        -------
        plotly.graph_objects.Figure
            Figura generada.
        """

        # Obtenemos el segmento visible.
        segment_data, segment_times, _ = self._get_segment(
            picks=picks,
            start=start,
            stop=stop,
        )

        # Calculamos media y desvio entre canales.
        mean_signal = np.mean(segment_data, axis=0)
        std_signal = np.std(segment_data, axis=0)

        # Calculamos bandas superior e inferior.
        upper = mean_signal + std_signal
        lower = mean_signal - std_signal

        # Creamos la figura.
        fig = go.Figure()

        # Banda inferior.
        fig.add_trace(
            go.Scatter(
                x=segment_times,
                y=lower,
                mode="lines",
                line=dict(width=0),
                name="Media - STD",
                showlegend=False,
            )
        )

        # Banda superior rellenando hasta la inferior.
        fig.add_trace(
            go.Scatter(
                x=segment_times,
                y=upper,
                mode="lines",
                fill="tonexty",
                name="Desvio estandar",
            )
        )

        # Curva central de la media.
        fig.add_trace(
            go.Scatter(
                x=segment_times,
                y=mean_signal,
                mode="lines",
                name="Media",
            )
        )

        # Configuramos layout.
        fig.update_layout(
            title=title,
            xaxis_title="Tiempo [s]",
            yaxis_title="Amplitud",
            hovermode="x unified",
        )

        # Slider temporal.
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="linear",
            )
        )

        return fig
    