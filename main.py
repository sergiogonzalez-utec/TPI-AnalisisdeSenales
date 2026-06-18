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

# Frecuencias de muestreo reales de cada archivo.
SFREQ_ECG = 200.0  # OpenBCI (cabecera: Sample Rate = 200 Hz)
SFREQ_EMG = 200.0  # OpenBCI (cabecera: Sample Rate = 200 Hz)
SFREQ_EEG = 1000.0 

# Cada entrada describe una senal a procesar y graficar.
# Formato:
#   "openbci"  -> TXT estilo OpenBCI (columna 0 = indice de muestra,
#                 columnas 1 a 4 = canales EXG).
#   "csv"      -> CSV generico separado por comas.
#   "espacios" -> TXT separado por espacios, con fila de encabezados
#                 (canales en las columnas indicadas).
#
# Tipo de senal: "raw", "eeg", "ecg" o "emg".

SENALES = [
    {
        "ruta": "docs/senal_ecg.txt",
        "formato": "openbci",
        "tipo_senal": "ecg",
        "sfreq": SFREQ_ECG,
        "columnas_canales": (1, 2, 3, 4),
        # Ventana de epoca centrada en el latido (complejo QRS).
        "tmin": -0.2,
        "tmax": 0.6,
        "anotaciones": [
            {"onset": 5.0, "duration": 1.0, "description": "Artefacto de movimiento"},
            {"onset": 15.0, "duration": 0.0, "description": "Extrasistole"},
            {"onset": 28.0, "duration": 2.0, "description": "Linea de base inestable"},
        ],
        # 6 epocas.
        "eventos": [
            {"onset": 8.0, "event_id": 1, "description": "Latido normal"},
            {"onset": 18.0, "event_id": 1, "description": "Latido normal"},
            {"onset": 25.0, "event_id": 1, "description": "Latido normal"},
            {"onset": 35.0, "event_id": 2, "description": "Latido anormal"},
            {"onset": 45.0, "event_id": 2, "description": "Latido anormal"},
            {"onset": 52.0, "event_id": 2, "description": "Latido anormal"},
        ],
    },
    {
        "ruta": "docs/senal_eeg.txt",
        "formato": "espacios",
        "tipo_senal": "eeg",
        "sfreq": SFREQ_EEG,
        # Archivo de 32 columnas (Fp1, Fp2, ... ECG).
        # Tomamos los 32 canales.
        "columnas_canales": tuple(range(32)),
        # Ventana de epoca tipo ERP (respuesta al estimulo).
        "tmin": -0.2,
        "tmax": 0.8,
        "anotaciones": [
            {"onset": 7.0, "duration": 0.5, "description": "Parpadeo"},
            {"onset": 22.0, "duration": 1.5, "description": "Artefacto muscular"},
            {"onset": 40.0, "duration": 0.0, "description": "Ojos abiertos"},
        ],
        # 8 epocas.
        "eventos": [
            {"onset": 10.0, "event_id": 1, "description": "Estimulo visual"},
            {"onset": 20.0, "event_id": 1, "description": "Estimulo visual"},
            {"onset": 30.0, "event_id": 1, "description": "Estimulo visual"},
            {"onset": 45.0, "event_id": 1, "description": "Estimulo visual"},
            {"onset": 55.0, "event_id": 2, "description": "Estimulo auditivo"},
            {"onset": 70.0, "event_id": 2, "description": "Estimulo auditivo"},
            {"onset": 90.0, "event_id": 2, "description": "Estimulo auditivo"},
            {"onset": 110.0, "event_id": 2, "description": "Estimulo auditivo"},
        ],
    },
    {
        "ruta": "docs/senal_emg.txt",
        "formato": "openbci",
        "tipo_senal": "emg",
        "sfreq": SFREQ_EMG,
        "columnas_canales": (1, 2, 3, 4),
        # Ventana de epoca corta alrededor de la contraccion.
        "tmin": -0.1,
        "tmax": 0.5,
        "anotaciones": [
            {"onset": 2.3, "duration": 0.9, "description": "Contraccion sostenida"},
            {"onset": 16.0, "duration": 0.0, "description": "Pico de fuerza"},
            {"onset": 30.0, "duration": 3.0, "description": "Fatiga muscular"},
        ],
        # 4 epocas.
        "eventos": [
            {"onset": 6.0, "event_id": 1, "description": "Contraccion"},
            {"onset": 14.0, "event_id": 1, "description": "Contraccion"},
            {"onset": 24.0, "event_id": 2, "description": "Reposo"},
            {"onset": 34.0, "event_id": 2, "description": "Reposo"},
        ],
    },
]


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

    formato = formato.lower() # Normalizamos el formato a minusculas para evitar errores por mayusculas.

    if formato == "openbci": # Leemos el formato de texto estilo OpenBCI, que tiene una estructura particular con comentarios y encabezados.
        return leer_openbci_txt(
            ruta=ruta,
            columnas_canales=columnas_canales,
        )

    if formato == "csv": # Leemos un archivo CSV generico, asumiendo que las columnas indicadas contienen los canales de interes.
        return leer_csv_generico(
            ruta=ruta,
            columnas_canales=columnas_canales,
        )

    if formato == "espacios": 
        return leer_txt_espacios(
            ruta=ruta,
            columnas_canales=columnas_canales,
        )

    raise ValueError("formato debe ser 'openbci', 'csv' o 'espacios'.") 


def leer_openbci_txt(ruta, columnas_canales):
    """
    Lee un archivo TXT estilo OpenBCI.
    """

    df = pd.read_csv( # Leemos el archivo con pandas, usando el caracter '%' como comentario para ignorar las lineas de encabezado y comentarios.
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


def leer_txt_espacios(ruta, columnas_canales):
    """
    Lee un archivo TXT separado por espacios con una fila
    de encabezados (nombres de canal) en la primera linea.
    """

    df = pd.read_csv(
        ruta,
        sep=r"\s+",
        header=0,
        skip_blank_lines=True,
    )

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
        # El pasabanda EMG por defecto llega a 150 Hz; lo limitamos
        # para no superar la frecuencia de Nyquist de la senal.
        nyquist = senal.sfreq / 2.0
        h_freq = min(150.0, nyquist * 0.9)
        return senal.filtrar(h_freq=h_freq)

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

def agregar_anotaciones(senal, anotaciones):
    """
    Agrega a la senal las anotaciones definidas en la config.
    """

    for anotacion in anotaciones:
        senal.anotaciones.add_annotation(
            onset=anotacion["onset"],
            duration=anotacion["duration"],
            description=anotacion["description"],
        )


def agregar_eventos(senal, eventos):
    """
    Agrega a la senal los eventos definidos en la config.
    """

    for evento in eventos:
        senal.eventos.add_event(
            onset=evento["onset"],
            event_id=evento["event_id"],
            description=evento["description"],
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

def procesar_senal(config, indice):
    """
    Procesa y grafica una senal segun su configuracion.
    """

    ruta = config["ruta"]
    formato = config["formato"]
    tipo_senal = config["tipo_senal"]
    sfreq = config["sfreq"]
    columnas_canales = config["columnas_canales"]
    anotaciones = config["anotaciones"]
    eventos = config["eventos"]
    tmin = config["tmin"]
    tmax = config["tmax"]

    print("\n" + "#" * 60)
    print(f"# PROCESANDO SENAL {tipo_senal.upper()} ({ruta})")
    print("#" * 60)

    data = leer_archivo(
        ruta=ruta,
        formato=formato,
        columnas_canales=columnas_canales,
    )

    senal = crear_senal(
        data=data,
        sfreq=sfreq,
        tipo_senal=tipo_senal,
    )

    agregar_anotaciones(senal, anotaciones)
    agregar_eventos(senal, eventos)

    senal_limpia = limpiar_senal(
        senal=senal,
        tipo_senal=tipo_senal,
    )

    epocas = Epocas(
        raw=senal_limpia,
        tmin=tmin,
        tmax=tmax,
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
        title=f"{tipo_senal.upper()} filtrada con anotaciones",
        show=False,
    )

    abrir_figura_html(
        fig=fig_raw,
        nombre_archivo=f"{indice:02d}_{tipo_senal}_senal_filtrada.html",
    )

    fig_epocas = plot_epochs(
        epocas,
        picks=senal_limpia.info.ch_names[:2],
        # Mostramos todas las epocas de la senal.
        max_epochs=epocas.get_data().shape[0],
        superpose=False,
        normalize=True,
        title=f"Epocas individuales - {tipo_senal.upper()}",
        show=False,
    )

    abrir_figura_html(
        fig=fig_epocas,
        nombre_archivo=f"{indice:02d}_{tipo_senal}_epocas_individuales.html",
    )


def main():
    for indice, config in enumerate(SENALES, start=1):
        procesar_senal(config=config, indice=indice)


if __name__ == "__main__":
    main()