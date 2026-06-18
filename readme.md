# TPI - Analisis de Senales (PDA 2026)

Libreria en Python para el **procesamiento, analisis y visualizacion** de senales
fisiologicas de **EEG, ECG y EMG**, desarrollada con Programacion Orientada a Objetos.

## Autores
- FLORIN Matias
- GONZALEZ Sergio

## Descripcion general

La libreria modela una senal fisiologica y su metadata mediante clases reutilizables:

- **`Info`**: almacena la metadata de la senal (nombres y tipos de canal, frecuencia
  de muestreo, canales malos, informacion del experimento, filtros, etc.).
- **`Anotaciones`** y **`Eventos`** (modulo `eventos`): marcas temporales continuas
  (onset, duracion, descripcion) y marcas puntuales (muestra, `event_id`).
- **`RawSignal`**: clase base para una senal continua `(n_canales, n_muestras)`.
- **`EEGSignal`, `ECGSignal`, `EMGSignal`**: heredan de `RawSignal` y agregan
  procesamiento especifico (filtrado, deteccion de picos R, RMS/envolvente, Hilbert, etc.).
- **`Epocas`** (modulo `epocas`): segmenta la senal en trials a partir de los eventos.
- Modulo **`visualizacion`**: graficos interactivos con Plotly (`PlotEngine`, `plot_raw`,
  `plot_epochs`).

## Instalacion

Se recomienda usar un entorno con **Python 3.10 o superior** (por ejemplo, con Miniconda).

```bash
# 1. (opcional) crear y activar un entorno
conda create -n senales python=3.10
conda activate senales

# 2. instalar la libreria en modo editable
pip install -e .

# 3. para desarrollo y testing (incluye pytest, black y ruff)
pip install -e ".[dev]"
```

## Estructura de carpetas

```
TPI-AnalisisdeSenales/
|-- pyproject.toml
|-- readme.md
|-- src/
|   |-- tpi_analisisdesenales/
|       |-- __init__.py
|       |-- info/              # clase Info (metadata)
|       |-- eventos/           # Anotaciones y Eventos
|       |-- signals/           # RawSignal, EEGSignal, ECGSignal, EMGSignal
|       |-- epocas/            # clase Epocas
|       |-- preprocesamiento/  # filtros (pasabanda, notch, Hilbert, etc.)
|       |-- visualizacion/     # PlotEngine, plot_raw, plot_epochs
|-- tests/                     # tests de pytest
|-- docs/
    |-- UML/                   # diagrama UML
```

## Como ejecutar los tests

Desde la raiz del proyecto:

```bash
python -m pytest
```

La configuracion de pytest (en `pyproject.toml`) ya activa el modo detallado, por lo que
se muestra el resultado `PASSED`/`FAILED` de cada test. Para una salida compacta:

```bash
python -m pytest -q
```

## Ejemplo minimo de uso

```python
import numpy as np

from tpi_analisisdesenales.info import Info
from tpi_analisisdesenales.signals import EEGSignal, ECGSignal
from tpi_analisisdesenales.eventos import Anotaciones

# ----- Senal EEG -----
data_eeg = np.random.randn(3, 1000)          # 3 canales, 1000 muestras
eeg = EEGSignal(
    data=data_eeg,
    sfreq=250,
    ch_names=["C3", "Cz", "C4"],
    ch_types=["eeg", "eeg", "eeg"],
)

print(eeg.describe())                         # estadisticas por canal
eeg.plot()                                    # grafico interactivo (Plotly)

# ----- Senal ECG con anotaciones -----
data_ecg = np.random.randn(1, 2000)
info = Info(ch_names=["DII"], sfreq=500, ch_types=["ecg"])
ecg = ECGSignal(data=data_ecg, info=info)

anot = Anotaciones()
anot.add(onset=1.0, duration=0.5, description="Artefacto")
ecg.set_anotaciones(anot)

picos_r = ecg.detectar_picos_r()              # deteccion de picos R
print("Picos R detectados:", len(picos_r))
```
