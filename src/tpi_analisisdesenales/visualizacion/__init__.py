"""
Subpaquete de visualizacion de la libreria.

Aqui se centralizan las herramientas para graficar señales,
anotaciones y, mas adelante, epocas u otras representaciones.
"""

from .plot_engine import PlotEngine
from .plot_raw import plot_raw
from .plot_epochs import plot_epochs, plot_epochs_average

__all__ = ["PlotEngine", "plot_raw", "plot_epochs", "plot_epochs_average"]

