"""
Paquete principal de la libreria TPI Analisis de Senales.

Este archivo permite exponer de forma centralizada las clases
mas importantes del proyecto para facilitar los imports.
"""

from .info import Info
from .anotaciones import Anotaciones
from .eventos import Eventos
from .raw_signal import RawSignal
from .eeg_signal import EEGSignal
from .ecg_signal import ECGSignal
from .emg_signal import EMGSignal
from .epocas import Epocas

__all__ = [
    "Info",
    "Anotaciones",
    "Eventos",
    "RawSignal",
    "EEGSignal",
    "ECGSignal",
    "EMGSignal",
    "Epocas",
]