# Diagrama UML - TPI Analisis de Senales

Diagrama de clases actual de la libreria. Se renderiza automaticamente en GitHub
y en VS Code (extension Markdown Preview Mermaid).

```mermaid
classDiagram
    direction TB

    class Info {
        -list _ch_names
        -float _sfreq
        -list _ch_types
        -list _bads
        -int _n_samples
        +str description
        +str experimenter
        +dict subject_info
        +line_freq
        +lowpass
        +highpass
        +list notch_freqs
        +ch_names() property
        +sfreq() property
        +ch_types() property
        +bads() property
        +n_channels() property
        +n_samples() property
        +duration() property
        +get(key, default)
        +keys()
        +items()
        +rename_channels(mapping)
        +add_bad(ch)
        +remove_bad(ch)
        +__contains__(key)
        +__getitem__(key)
        +__len__()
        +__str__()
    }

    class Anotaciones {
        -list _anotaciones
        +add(onset, duration, description)
        +add_annotation(...)
        +find(text)
        +get_annotations() DataFrame
        +to_list()
        +clear()
        +__iter__()
        +__getitem__(i)
        +__len__()
        +__repr__()
    }

    class Eventos {
        +float sfreq
        -list _eventos
        +add_event(onset, event_id, description)
        +add_event_sample(sample, event_id, description)
        +rename_event_id(old_id, new_id)
        +find(text)
        +get_events() DataFrame
        +to_list()
        +clear()
        +__iter__()
        +__getitem__(i)
        +__len__()
        +__repr__()
    }

    class RawSignal {
        +ndarray data
        +Info info
        +Eventos eventos
        +Anotaciones anotaciones
        +int first_samp
        +sfreq() property
        +duration() property
        +get_data(picks, start, stop, times, reject)
        +describe() DataFrame
        +resumen() dict
        +drop_channels(ch_names)
        +crop(tmin, tmax)
        +get_channel(ch)
        +pick_channels(picks)
        +pick_types(**kwargs)
        +set_anotaciones(anotaciones)
        +add_annotation(...)
        +plot(...)
        +__getitem__(item)
        +__str__()
    }

    class EEGSignal {
        +str tipo_senal
        +ndarray times
        +dict montage
        +reference
        +str units
        +dict event_id
        +meas_date
        +list filters
        +bool is_epoched
        +bool is_filtered
        +n_channels() property
        +n_samples() property
        +n_trials() property
        +get_data(...)
        +get_channels(picks)
        +crop(tmin, tmax)
        +set_reference(ch)
        +drop_channels(picks)
        +filter(l_freq, h_freq)
        +filtrar(...)
        +notch_filter(...)
        +aplicar_hilbert()
        +frecuencia_instantanea()
        +get_epochs(...)
        +describe() DataFrame
        +describe_eeg()
        +copy()
        +plot(...)
    }

    class ECGSignal {
        +str tipo_senal
        +ndarray r_peaks
        +float heart_rate
        +filtrar(...)
        +detectar_picos_r(canal, distancia_minima_s)
        +calcular_frecuencia_cardiaca(canal)
        +describe_ecg(canal)
    }

    class EMGSignal {
        +str tipo_senal
        +str muscle_group
        +int sampling_window
        +list filters_applied
        +dict features
        +validate_structure()
        +filtrar(...)
        +rectificar()
        +envelope(rectify)
        +rms_per_channel()
        +mav_per_channel()
        +variance_per_channel()
        +describe() DataFrame
        +feature_extraction()
        +describe_emg()
    }

    class Epocas {
        +RawSignal raw
        +Eventos eventos
        +float tmin
        +float tmax
        +event_id
        +float sfreq
        +list ch_names
        +ndarray data
        +ndarray times
        +list metadata
        +get_data()
        +get_metadata() DataFrame
        +average()
        +select_event_id(event_id)
        +__len__()
        +__getitem__(i)
        +__repr__()
    }

    class PlotEngine {
        +ndarray data
        +float sfreq
        +list ch_names
        +anotaciones
        +ndarray times
        +plot_signals(...)
        +plot_mean_std(...)
    }

    %% Herencia
    RawSignal <|-- EEGSignal
    RawSignal <|-- ECGSignal
    RawSignal <|-- EMGSignal

    %% Composicion / asociacion
    RawSignal *-- Info : info
    RawSignal o-- Anotaciones : anotaciones
    RawSignal o-- Eventos : eventos

    Epocas --> RawSignal : segmenta
    Epocas --> Eventos : usa

    RawSignal ..> PlotEngine : grafica via plot_raw
    PlotEngine ..> Anotaciones : dibuja
```

## Relaciones

- **Herencia:** `EEGSignal`, `ECGSignal` y `EMGSignal` heredan de `RawSignal`.
- **Composicion:** `RawSignal` contiene un objeto `Info` (no existe sin metadata).
- **Agregacion:** `RawSignal` tiene `Anotaciones` y `Eventos` (pueden ser opcionales).
- **Dependencia:** `Epocas` segmenta una senal (`RawSignal`) a partir de sus `Eventos`;
  la visualizacion (`PlotEngine`) recibe los datos y las anotaciones para graficar.
