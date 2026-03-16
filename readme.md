# 🎬 From Video to Text - on your PC

## Il problema

Hai ore di registrazioni di chiamate, riunioni, lezioni da recuperare e sintetizzare — ma non hai il tempo di riascoltarle una per una.

Non parliamo di video online come YouTube: per quelli esistono già tantissimi tool in grado di trascrivere e creare chatbot interrogabili sulle fonti.

Parliamo di **registrazioni video e audio in locale**, sul tuo device. File che restano nei tuoi hard disk, nei NAS aziendali, nelle cartelle condivise. Come trasformare GB di video in **testo ricercabile e utilizzabile** dai tuoi strumenti AI — aziendali e non?

## La soluzione

Ho creato un processo **interamente in locale**, open source, che estrae e trascrive il contenuto audio di qualsiasi registrazione video — senza inviare nulla al cloud, senza abbonamenti, senza limiti di utilizzo.

Un'applicazione desktop con interfaccia grafica moderna che **estrae l'audio**, lo **segmenta** in parti di dimensione configurabile e lo **trascrive automaticamente** usando modelli Whisper. Supporta **italiano** e **inglese** con rilevamento automatico della lingua.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

---

## ✨ Funzionalità

- **Estrazione + segmentazione audio** da qualsiasi formato video (MP4, AVI, MKV, MOV, ecc.) con dimensione segmenti configurabile
- **Trascrizione locale** con [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — modello consigliato `large-v3-turbo`, accurato su italiano e inglese
- **Filtro VAD** integrato contro le allucinazioni nei silenzi
- **Output flessibile**: testo semplice o dettagliato con timestamp, con report completo delle metriche
- **Interfaccia moderna** con CustomTkinter (dark/light) + modalità CLI standalone (`transcriber.py`)

---

## 📋 Prerequisiti

### 1. Python 3.8 o superiore

Verifica se Python è già installato aprendo un terminale:

```bash
python --version
```

Se non è installato:

- **Windows**: scarica da [python.org/downloads](https://www.python.org/downloads/). Durante l'installazione **spunta "Add Python to PATH"**.
- **macOS**: `brew install python` oppure scarica da [python.org](https://www.python.org/downloads/)
- **Linux (Ubuntu/Debian)**: `sudo apt update && sudo apt install python3 python3-pip python3-venv`

### 2. FFmpeg

FFmpeg è necessario per estrarre e segmentare l'audio dai video. Verifica se è installato:

```bash
ffmpeg -version
```

Se non è installato:

<details>
<summary><b>🪟 Windows</b></summary>

**Opzione A — winget (consigliata, Windows 10/11):**
```bash
winget install FFmpeg
```
Riavvia il terminale dopo l'installazione.

**Opzione B — Installazione manuale:**
1. Vai su [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/) e scarica `ffmpeg-release-essentials.zip`
2. Estrai lo ZIP in una cartella, es. `C:\ffmpeg`
3. Aggiungi `C:\ffmpeg\bin` al PATH di sistema:
   - Cerca "Variabili d'ambiente" nel menu Start
   - Modifica la variabile `Path` → Aggiungi `C:\ffmpeg\bin`
4. Riavvia il terminale e verifica con `ffmpeg -version`
</details>

<details>
<summary><b>🍎 macOS</b></summary>

```bash
brew install ffmpeg
```
Se non hai Homebrew: [brew.sh](https://brew.sh/)
</details>

<details>
<summary><b>🐧 Linux (Ubuntu/Debian)</b></summary>

```bash
sudo apt update
sudo apt install ffmpeg
```
</details>

### 3. Git (per clonare il repository)

```bash
git --version
```

Se non è installato:
- **Windows**: scarica da [git-scm.com](https://git-scm.com/)
- **macOS**: `brew install git` oppure `xcode-select --install`
- **Linux**: `sudo apt install git`

---

## 🚀 Installazione del progetto

```bash
# 1. Clona il repository
git clone https://github.com/PierpaoloPalmiotti/from_video_to_transcript.git
cd from_video_to_transcript

# 2. Crea un virtual environment (consigliato)
python -m venv venv

# 3. Attiva il virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Installa le dipendenze
pip install -r requirements.txt
```

### ⚠️ Nota sul primo avvio

Al primo avvio della trascrizione, il modello Whisper scelto verrà **scaricato automaticamente** da Hugging Face. Le dimensioni variano in base al modello:

| Modello | Download |
|---|---|
| `tiny` | ~75 MB |
| `small` | ~460 MB |
| `medium` | ~1.5 GB |
| `large-v3-turbo` | ~1.6 GB |
| `large-v3` | ~3 GB |

Il download avviene una sola volta. Le esecuzioni successive useranno la cache locale.

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

## ⚡ Performance reali e proiezioni

### Benchmark misurato

Dai test effettuati con `large-v3-turbo` in quantizzazione `int8` su CPU, il tempo end-to-end (splitting + trascrizione) è di circa:

> **~20 sec/MB** di video sorgente

Questo valore include il caricamento del modello, l'estrazione audio, la segmentazione e la trascrizione completa.

### Proiezioni tempi E2E per dimensione video

| Dimensione video | Tempo stimato E2E | Note |
|---|---|---|
| **50 MB** | ~17 min | Clip breve, video compresso |
| **100 MB** | ~33 min | Video medio di qualche minuto |
| **200 MB** | ~1h 7min | Registrazione ~15-30 min |
| **500 MB** | ~2h 47min | Lezione / meeting ~1h |
| **1 GB** | ~5h 41min | Conferenza lunga / webinar |
| **2 GB** | ~11h 22min | Evento / registrazione estesa |
| **5 GB** | ~28h 24min | Consigliato usare GPU |

> ⚠️ I tempi sono indicativi e dipendono da CPU, RAM disponibile, e complessità dell'audio (più parlato = più lavoro per il modello). Con una **GPU NVIDIA** (RTX 3060+) i tempi di trascrizione si riducono di **5-8x**, portando il totale a circa **3-5 sec/MB**.

### Confronto CPU vs GPU vs Apple Silicon

| Setup | sec/MB | 200 MB | 1 GB |
|---|---|---|---|
| CPU (i7/Ryzen 7, int8) | ~20 | ~1h 7min | ~5h 41min |
| CPU (i9/Ryzen 9, int8) | ~14 | ~47 min | ~4h |
| MacBook Air M2/M3 (int8) | ~16 | ~53 min | ~4h 33min |
| MacBook Pro M2 Pro/M3 Pro (int8) | ~12 | ~40 min | ~3h 25min |
| MacBook Pro M3 Max/M4 Pro (int8) | ~9 | ~30 min | ~2h 34min |
| GPU (RTX 3060, float16) | ~4 | ~13 min | ~1h 8min |
| GPU (RTX 4090, float16) | ~2 | ~7 min | ~34 min |

> 💡 **Nota su macOS**: faster-whisper su Apple Silicon gira su CPU (non sfrutta Metal/GPU nativamente), ma le performance dei chip M-series sono eccellenti grazie alla bandwidth di memoria unificata e all'efficienza dei core. Un MacBook Pro M3 Pro si colloca a metà tra un i9 desktop e una RTX 3060. Per sfruttare appieno la GPU Apple, valuta alternative come [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) che supportano l'accelerazione Metal nativa.

> 💡 **Consiglio**: per video superiori a 500 MB, una GPU dedicata (o un Mac con chip Pro/Max) fa una differenza enorme. Se lavori solo su CPU, puoi lanciare la trascrizione di notte su file grandi.

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

## 🔧 Troubleshooting

| Problema | Soluzione |
|---|---|
| `FFmpeg non trovato` | Verifica che `ffmpeg -version` funzioni nel terminale. Se no, reinstallalo e riavvia il terminale. |
| `faster-whisper non installato` | Esegui `pip install faster-whisper` con il virtual environment attivo |
| Trascrizione lenta | Prova un modello più piccolo (`medium`, `small`) oppure usa una GPU con `--device cuda` |
| Il modello non si scarica | Verifica la connessione internet. La cache è in `~/.cache/huggingface/` |
| Errore CUDA / GPU | Assicurati di avere i driver NVIDIA aggiornati e CUDA toolkit installato |
| Allucinazioni nel testo | Il filtro VAD è attivo di default. Se persistono, prova `--lingua it` per forzare la lingua |

---

## 📄 Licenza

Questo progetto è distribuito sotto licenza [MIT](LICENSE).