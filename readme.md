# 🎬 Audio Splitter & Transcriber

Applicazione desktop con interfaccia grafica moderna per **estrarre audio da video**, **segmentarlo** in parti di dimensione configurabile e **trascriverlo automaticamente** in locale usando modelli Whisper.

Supporta **italiano** e **inglese** con rilevamento automatico della lingua.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

---

## ✨ Funzionalità

- **Estrazione audio** da qualsiasi formato video (MP4, AVI, MKV, MOV, ecc.)
- **Segmentazione** dell'audio in parti con dimensione massima configurabile (in MB)
- **Trascrizione locale** con [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — nessun dato inviato al cloud
- **Modello consigliato**: `large-v3-turbo` — veloce e accurato su italiano/inglese
- **Filtro VAD** integrato per evitare allucinazioni durante i silenzi
- **Due formati di output**: testo semplice o dettagliato con timestamp
- **Report completo** con tempi di splitting, trascrizione e metriche sec/MB
- **Interfaccia moderna** con CustomTkinter (tema dark/light)
- **Modalità standalone**: lo script `transcriber.py` funziona anche da riga di comando

---

## 📋 Requisiti

- **Python 3.8+**
- **FFmpeg** installato e nel PATH di sistema
- **RAM**: almeno 4 GB liberi per il modello `large-v3-turbo` (quantizzazione int8)
- **GPU** (opzionale): NVIDIA con CUDA per trascrizioni più veloci

### Installare FFmpeg

- **Windows**: scarica da [ffmpeg.org](https://ffmpeg.org/download.html) e aggiungi al PATH, oppure `winget install FFmpeg`
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

---

## 🚀 Installazione

```bash
# Clona il repository
git clone https://github.com/TUO_USERNAME/audio-splitter-transcriber.git
cd audio-splitter-transcriber

# Crea un virtual environment (consigliato)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Installa le dipendenze
pip install -r requirements.txt
```

> **Nota**: al primo avvio della trascrizione, il modello Whisper verrà scaricato automaticamente da Hugging Face (~1.5 GB per `large-v3-turbo`). Le esecuzioni successive saranno molto più veloci.

---

## 💻 Utilizzo

### Interfaccia grafica (GUI)

```bash
python main.py
```

1. Clicca **📁 Sfoglia** e seleziona un video
2. Imposta la dimensione massima per segmento (in MB)
3. Clicca **▶ Elabora Video** — i segmenti verranno salvati in una cartella con il nome del video
4. Clicca **🎙 Trascrivi Segmenti** — la trascrizione verrà salvata nella stessa cartella

Puoi anche usare **📂 Trascrivi da Cartella...** per trascrivere file audio già esistenti senza passare dallo splitting.

### Riga di comando (CLI)

```bash
# Trascrivi una cartella di file audio
python transcriber.py /percorso/cartella_audio/

# Forza italiano e formato dettagliato con timestamp
python transcriber.py /percorso/cartella_audio/ --lingua it --formato dettagliato

# Usa un modello più leggero se la RAM scarseggia
python transcriber.py /percorso/cartella_audio/ --modello medium

# Specifica un file di output personalizzato
python transcriber.py /percorso/cartella_audio/ --output risultato.txt
```

**Opzioni CLI disponibili:**

| Opzione | Default | Descrizione |
|---|---|---|
| `--modello` | `large-v3-turbo` | Modello Whisper da usare |
| `--lingua` | auto-detect | Codice lingua (`it`, `en`, ecc.) |
| `--device` | `auto` | `cpu`, `cuda` o `auto` |
| `--beam-size` | `5` | Beam search size (1 = veloce, 5 = accurato) |
| `--formato` | `txt` | `txt` o `dettagliato` |
| `--output` | `trascrizione.txt` | Percorso file di output |
| `--no-vad` | disattivato | Disabilita il filtro anti-allucinazioni |

---

## 📁 Struttura output

Ogni video genera una cartella dedicata con il proprio nome:

```
cartella_video/
├── intervista.mp4
├── intervista/
│   ├── intervista_segmento_1.wav
│   ├── intervista_segmento_2.wav
│   ├── intervista_segmento_3.wav
│   └── trascrizione.txt
├── riunione.mp4
└── riunione/
    ├── riunione_segmento_1.wav
    └── trascrizione.txt
```

---

## ⚡ Performance indicative

Con `large-v3-turbo` in quantizzazione `int8` su CPU:

| Hardware | 10 min di audio | 1 ora di audio |
|---|---|---|
| Laptop (i7 / Ryzen 7) | ~2.5-4 min | ~15-25 min |
| Desktop (i9 / Ryzen 9) | ~1.5-2.5 min | ~10-15 min |
| GPU (RTX 3060+) float16 | ~30-50 sec | ~3-5 min |

---

## 🛠 Scelta del modello

| Modello | RAM (int8) | Velocità | Qualità IT/EN |
|---|---|---|---|
| `tiny` | ~0.5 GB | ⚡⚡⚡⚡⚡ | ⭐⭐ |
| `small` | ~1 GB | ⚡⚡⚡⚡ | ⭐⭐⭐ |
| `medium` | ~2 GB | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| `large-v3-turbo` | ~2.5 GB | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ |
| `large-v3` | ~4 GB | ⚡ | ⭐⭐⭐⭐⭐ |

> **Consiglio**: usa `large-v3-turbo` come default. Scala a `medium` se la RAM non basta. Evita `tiny` e `base` per l'italiano.

---

## 📄 Licenza

Questo progetto è distribuito sotto licenza [MIT](LICENSE).