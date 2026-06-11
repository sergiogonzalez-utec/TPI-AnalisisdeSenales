from pathlib import Path
import webbrowser

import pandas as pd

from tpi_analisisdesenales.info import Info
from tpi_analisisdesenales.signals import RawSignal

from tpi_analisisdesenales.preprocesamiento.filtros import (
    filtro_notch,
    filtro_pasaaltos,
    filtro_pasabajos,
)

from tpi_analisisdesenales.epocas import Epocas

from tpi_analisisdesenales.visualizacion.plot_raw import plot_raw
from tpi_analisisdesenales.visualizacion.plot_epochs import (
    plot_epochs,
    plot_epochs_average,
)


# ============================================================
# CONFIGURACION
# ============================================================

RUTA_ARCHIVO = "docs/senal.txt"
SFREQ = 250.0


# ============================================================
# LECTURA DEL ARCHIVO
# ============================================================

def leer_openbci_txt(ruta, sfreq=250.0):
    """
    Lee un archivo TXT estilo OpenBCI y devuelve un RawSignal.
    """

    ruta = Path(ruta)

    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {ruta}")

    df = pd.read_csv(
        ruta,
        header=None,
        comment="%",
        skip_blank_lines=True,
    )

    # Convierte columnas a numero.
    # Las filas de encabezado quedan como NaN.
    df = df.apply(pd.to_numeric, errors="coerce")

    # Elimina filas que no tengan indice de muestra numerico.
    df = df.dropna(subset=[0])

    # Columnas 1 a 4: canales EXG principales.
    data = df.iloc[:, 1:5].to_numpy(dtype=float)

    # La libreria usa: n_canales x n_muestras.
    data = data.T

    info = Info(
        ch_names=["Canal 1", "Canal 2", "Canal 3", "Canal 4"],
        sfreq=sfreq,
        ch_types=["EEG", "EEG", "EEG", "EEG"],
        subject_info={"id": "EEG_test"},
    )

    raw = RawSignal(
        data=data,
        info=info,
    )

    return raw


# ============================================================
# FILTRADO
# ============================================================

def limpiar_raw(raw):
    """
    Aplica una cadena basica de limpieza.
    """

    data_limpia = filtro_notch(
        raw.data,
        raw.sfreq,
        freq=50.0,
        q=30.0,
    )

    data_limpia = filtro_pasaaltos(
        data_limpia,
        raw.sfreq,
        l_freq=0.5,
        order=4,
    )

    data_limpia = filtro_pasabajos(
        data_limpia,
        raw.sfreq,
        h_freq=40.0,
        order=4,
    )

    raw_limpia = RawSignal(
        data=data_limpia,
        info=raw.info,
        anotaciones=raw.anotaciones,
        eventos=raw.eventos,
    )

    return raw_limpia


# ============================================================
# ANOTACIONES Y EVENTOS
# ============================================================

def agregar_anotaciones_de_prueba(raw):
    """
    Agrega anotaciones visibles en la senal continua.
    """

    raw.anotaciones.add_annotation(
        onset=6.0,
        duration=1.5,
        description="Artefacto",
    )

    raw.anotaciones.add_annotation(
        onset=18.0,
        duration=0.0,
        description="Evento puntual",
    )

    raw.anotaciones.add_annotation(
        onset=32.0,
        duration=2.0,
        description="Movimiento",
    )


def agregar_eventos_de_prueba(raw):
    """
    Agrega eventos para crear epocas.
    """

    raw.eventos.add_event(
        onset=10.0,
        event_id=1,
        description="Estimulo A",
    )

    raw.eventos.add_event(
        onset=20.0,
        event_id=1,
        description="Estimulo A",
    )

    raw.eventos.add_event(
        onset=30.0,
        event_id=2,
        description="Estimulo B",
    )

    raw.eventos.add_event(
        onset=40.0,
        event_id=2,
        description="Estimulo B",
    )


# ============================================================
# GRAFICAS
# ============================================================

def abrir_figura_html(fig, nombre_archivo):
    """
    Guarda una figura Plotly como HTML y la abre en el navegador.
    """

    salida = Path(nombre_archivo).resolve()

    print(f"Guardando grafica: {salida}")
    print("Cantidad de trazas:", len(fig.data))

    fig.write_html(str(salida))
    webbrowser.open(salida.as_uri())


# ============================================================
# RESUMEN
# ============================================================

def mostrar_resumen(raw, raw_limpia, epocas):
    """
    Muestra informacion basica por consola.
    """

    print("===== RAW ORIGINAL =====")
    print("Data:", raw.data.shape)
    print("Frecuencia:", raw.sfreq)
    print("Canales:", raw.info.ch_names)

    print("\n===== RAW FILTRADA =====")
    print("Data:", raw_limpia.data.shape)
    print("Anotaciones:", len(raw_limpia.anotaciones))
    print("Eventos:", len(raw_limpia.eventos))

    print("\n===== EPOCAS =====")
    print(epocas)
    print("Data epocas:", epocas.get_data().shape)
    print("Promedio:", epocas.average().shape)
    print("Metadata:")
    print(epocas.get_metadata())


# ============================================================
# MAIN
# ============================================================

def main():
    raw = leer_openbci_txt(
        ruta=RUTA_ARCHIVO,
        sfreq=SFREQ,
    )

    agregar_anotaciones_de_prueba(raw)
    agregar_eventos_de_prueba(raw)

    raw_limpia = limpiar_raw(raw)

    epocas = Epocas(
        raw=raw_limpia,
        tmin=-0.2,
        tmax=0.8,
    )

    mostrar_resumen(
        raw=raw,
        raw_limpia=raw_limpia,
        epocas=epocas,
    )

    fig_raw = plot_raw(
        raw_limpia,
        start=0.0,
        stop=20.0,
        superpose=False,
        show_annotations=True,
        fill_annotations=True,
        normalize=True,
        title="Senal filtrada con anotaciones",
        show=False,
    )

    abrir_figura_html(
        fig=fig_raw,
        nombre_archivo="01_senal_filtrada.html",
    )

    input("Presiona Enter para mostrar las epocas individuales...")

    fig_epocas = plot_epochs(
        epocas,
        picks=["Canal 1", "Canal 2"],
        max_epochs=5,
        superpose=False,
        normalize=True,
        title="Epocas individuales",
        show=False,
    )

    abrir_figura_html(
        fig=fig_epocas,
        nombre_archivo="02_epocas_individuales.html",
    )

    input("Presiona Enter para mostrar el promedio de epocas...")

    fig_promedio = plot_epochs_average(
        epocas,
        picks=["Canal 1", "Canal 2"],
        normalize=True,
        title="Promedio de epocas",
        show=False,
    )

    abrir_figura_html(
        fig=fig_promedio,
        nombre_archivo="03_promedio_epocas.html",
    )


if __name__ == "__main__":
    main()