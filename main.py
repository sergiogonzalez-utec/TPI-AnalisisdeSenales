from pathlib import Path
import webbrowser

import pandas as pd

from tpi_analisisdesenales.info import Info

from tpi_analisisdesenales.signals.raw_signal import RawSignal
from tpi_analisisdesenales.signals.eeg_signal import EEGSignal
from tpi_analisisdesenales.signals.ecg_signal import ECGSignal
from tpi_analisisdesenales.signals.emg_signal import EMGSignal

from tpi_analisisdesenales.eventos import Anotaciones, Eventos
from tpi_analisisdesenales.epocas import Epocas

from tpi_analisisdesenales.preprocesamiento.filtros import (
    filtro_notch,
    filtro_pasaaltos,
    filtro_pasabajos,
)

from tpi_analisisdesenales.visualizacion.plot_raw import plot_raw
from tpi_analisisdesenales.visualizacion.plot_epochs import plot_epochs


# ============================================================
# CONFIGURACION
# ============================================================

RUTA_ARCHIVO = "docs/senal.txt"

FORMATO_ARCHIVO = "openbci"
# Opciones:
# "openbci"
# "csv"

TIPO_SENAL = "eeg"
# Opciones:
# "raw"
# "eeg"
# "ecg"
# "emg"

SFREQ = 250.0

COLUMNAS_CANALES = (1, 2, 3, 4)
# OpenBCI:
# columna 0: indice de muestra
# columnas 1 a 4: canales EXG
#
# CSV generico con solo canales:
# COLUMNAS_CANALES = (0, 1, 2, 3)


# ============================================================
# LECTURA DE ARCHIVOS
# ============================================================

def leer_archivo(ruta, formato, columnas_canales):
    """
    Lee un archivo y devuelve data con forma:
    n_canales x n_muestras.
    """

    ruta = Path(ruta)

    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {ruta}")

    formato = formato.lower()

    if formato == "openbci":
        return leer_openbci_txt(
            ruta=ruta,
            columnas_canales=columnas_canales,
        )

    if formato == "csv":
        return leer_csv_generico(
            ruta=ruta,
            columnas_canales=columnas_canales,
        )

    raise ValueError("formato debe ser 'openbci' o 'csv'.")


def leer_openbci_txt(ruta, columnas_canales):
    """
    Lee un archivo TXT estilo OpenBCI.
    """

    df = pd.read_csv(
        ruta,
        header=None,
        comment="%",
        skip_blank_lines=True,
    )

    # Convierte todo lo posible a numero.
    # La fila de encabezados queda como NaN y se elimina.
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=[0])

    data = df.iloc[:, list(columnas_canales)].to_numpy(dtype=float)

    # La libreria usa n_canales x n_muestras.
    data = data.T

    return data


def leer_csv_generico(ruta, columnas_canales):
    """
    Lee un archivo CSV generico.
    """

    df = pd.read_csv(ruta)

    data = df.iloc[:, list(columnas_canales)].to_numpy(dtype=float)

    # La libreria usa n_canales x n_muestras.
    data = data.T

    return data


# ============================================================
# CREACION DE INFO Y SENAL
# ============================================================

def crear_nombres_y_tipos(data, tipo_senal):
    """
    Crea nombres y tipos de canal segun el tipo de senal.
    """

    n_canales = data.shape[0]
    tipo_senal = tipo_senal.lower()

    if tipo_senal == "eeg":
        ch_names = [f"EEG {i + 1}" for i in range(n_canales)]
        ch_types = ["EEG"] * n_canales

    elif tipo_senal == "ecg":
        ch_names = [f"ECG {i + 1}" for i in range(n_canales)]
        ch_types = ["ECG"] * n_canales

    elif tipo_senal == "emg":
        ch_names = [f"EMG {i + 1}" for i in range(n_canales)]
        ch_types = ["EMG"] * n_canales

    else:
        ch_names = [f"Canal {i + 1}" for i in range(n_canales)]
        ch_types = ["RAW"] * n_canales

    return ch_names, ch_types


def crear_info(data, sfreq, tipo_senal):
    """
    Crea un objeto Info.
    """

    ch_names, ch_types = crear_nombres_y_tipos(
        data=data,
        tipo_senal=tipo_senal,
    )

    info = Info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=ch_types,
        subject_info={"id": "test_signal"},
    )

    return info


def crear_senal(data, sfreq, tipo_senal):
    """
    Crea RawSignal, EEGSignal, ECGSignal o EMGSignal.
    """

    tipo_senal = tipo_senal.lower()

    ch_names, ch_types = crear_nombres_y_tipos(
        data=data,
        tipo_senal=tipo_senal,
    )

    anotaciones = Anotaciones()
    eventos = Eventos(sfreq=sfreq)

    if tipo_senal == "raw":
        info = crear_info(
            data=data,
            sfreq=sfreq,
            tipo_senal=tipo_senal,
        )

        return RawSignal(
            data=data,
            info=info,
            anotaciones=anotaciones,
            eventos=eventos,
        )

    if tipo_senal == "eeg":
        return EEGSignal(
            data=data,
            sfreq=sfreq,
            ch_names=ch_names,
            ch_types=ch_types,
            anotaciones=anotaciones,
            eventos=eventos,
            subject_info={"id": "test_signal"},
        )

    if tipo_senal == "ecg":
        info = crear_info(
            data=data,
            sfreq=sfreq,
            tipo_senal=tipo_senal,
        )

        return ECGSignal(
            data=data,
            info=info,
            anotaciones=anotaciones,
            eventos=eventos,
        )

    if tipo_senal == "emg":
        info = crear_info(
            data=data,
            sfreq=sfreq,
            tipo_senal=tipo_senal,
        )

        return EMGSignal(
            data=data,
            info=info,
            anotaciones=anotaciones,
            eventos=eventos,
        )

    raise ValueError("tipo_senal debe ser 'raw', 'eeg', 'ecg' o 'emg'.")


# ============================================================
# FILTRADO
# ============================================================

def filtrar_data_general(data, sfreq):
    """
    Aplica filtros generales:
    notch 50 Hz, pasa altos 0.5 Hz y pasa bajos 40 Hz.
    """

    data_limpia = filtro_notch(
        data,
        sfreq,
        freq=50.0,
        q=30.0,
    )

    data_limpia = filtro_pasaaltos(
        data_limpia,
        sfreq,
        l_freq=0.5,
        order=4,
    )

    data_limpia = filtro_pasabajos(
        data_limpia,
        sfreq,
        h_freq=40.0,
        order=4,
    )

    return data_limpia


def limpiar_senal(senal, tipo_senal):
    """
    Limpia la senal segun su tipo.
    """

    tipo_senal = tipo_senal.lower()

    if tipo_senal == "eeg":
        return senal.filtrar()

    if tipo_senal == "ecg":
        return senal.filtrar()

    if tipo_senal == "emg":
        return senal.filtrar()

    data_limpia = filtrar_data_general(
        data=senal.data,
        sfreq=senal.sfreq,
    )

    return RawSignal(
        data=data_limpia,
        info=senal.info,
        anotaciones=senal.anotaciones,
        eventos=senal.eventos,
        first_samp=senal.first_samp,
    )


# ============================================================
# ANOTACIONES Y EVENTOS
# ============================================================

def agregar_anotaciones_de_prueba(senal):
    """
    Agrega anotaciones visibles en la senal continua.
    """

    senal.anotaciones.add_annotation(
        onset=6.0,
        duration=1.5,
        description="Artefacto",
    )

    senal.anotaciones.add_annotation(
        onset=18.0,
        duration=0.0,
        description="Evento puntual",
    )

    senal.anotaciones.add_annotation(
        onset=32.0,
        duration=2.0,
        description="Movimiento",
    )


def agregar_eventos_de_prueba(senal):
    """
    Agrega eventos para crear epocas.
    """

    senal.eventos.add_event(
        onset=10.0,
        event_id=1,
        description="Estimulo A",
    )

    senal.eventos.add_event(
        onset=20.0,
        event_id=1,
        description="Estimulo A",
    )

    senal.eventos.add_event(
        onset=30.0,
        event_id=2,
        description="Estimulo B",
    )

    senal.eventos.add_event(
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

def mostrar_resumen(senal, senal_limpia, epocas):
    """
    Muestra informacion basica por consola.
    """

    print("===== SENAL ORIGINAL =====")
    print("Clase:", type(senal).__name__)
    print("Data:", senal.data.shape)
    print("Frecuencia:", senal.sfreq)
    print("Canales:", senal.info.ch_names)

    print("\n===== SENAL FILTRADA =====")
    print("Clase:", type(senal_limpia).__name__)
    print("Data:", senal_limpia.data.shape)
    print("Anotaciones:", len(senal_limpia.anotaciones))
    print("Eventos:", len(senal_limpia.eventos))

    if isinstance(senal_limpia, EEGSignal):
        print("\n===== RESUMEN EEG =====")
        print(senal_limpia.describe_eeg())

    if isinstance(senal_limpia, ECGSignal):
        try:
            print("\n===== RESUMEN ECG =====")
            print(senal_limpia.describe_ecg(canal=0))
        except ValueError as error:
            print("\nNo se pudo calcular el resumen ECG:")
            print(error)

    if isinstance(senal_limpia, EMGSignal):
        print("\n===== RESUMEN EMG =====")
        print(senal_limpia.describe_emg())

    print("\n===== EPOCAS =====")
    print(epocas)
    print("Data epocas:", epocas.get_data().shape)
    print("Metadata:")
    print(epocas.get_metadata())


# ============================================================
# MAIN
# ============================================================

def main():
    data = leer_archivo(
        ruta=RUTA_ARCHIVO,
        formato=FORMATO_ARCHIVO,
        columnas_canales=COLUMNAS_CANALES,
    )

    senal = crear_senal(
        data=data,
        sfreq=SFREQ,
        tipo_senal=TIPO_SENAL,
    )

    agregar_anotaciones_de_prueba(senal)
    agregar_eventos_de_prueba(senal)

    senal_limpia = limpiar_senal(
        senal=senal,
        tipo_senal=TIPO_SENAL,
    )

    epocas = Epocas(
        raw=senal_limpia,
        tmin=-0.2,
        tmax=0.8,
    )

    mostrar_resumen(
        senal=senal,
        senal_limpia=senal_limpia,
        epocas=epocas,
    )

    fig_raw = plot_raw(
        senal_limpia,
        start=0.0,
        stop=20.0,
        superpose=False,
        show_annotations=True,
        fill_annotations=True,
        normalize=True,
        title=f"{TIPO_SENAL.upper()} filtrada con anotaciones",
        show=False,
    )

    abrir_figura_html(
        fig=fig_raw,
        nombre_archivo="01_senal_filtrada.html",
    )

    input("Presiona Enter para mostrar las epocas individuales...")

    fig_epocas = plot_epochs(
        epocas,
        picks=senal_limpia.info.ch_names[:2],
        max_epochs=5,
        superpose=False,
        normalize=True,
        title=f"Epocas individuales - {TIPO_SENAL.upper()}",
        show=False,
    )

    abrir_figura_html(
        fig=fig_epocas,
        nombre_archivo="02_epocas_individuales.html",
    )


if __name__ == "__main__":
    main()