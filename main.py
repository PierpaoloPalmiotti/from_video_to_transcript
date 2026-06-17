import customtkinter as ctk
from tkinter import messagebox, filedialog
import os
import sys
import re
import gc
import math
import time
import json
import warnings
import subprocess
import tempfile
import threading
from datetime import datetime

# Sopprimi i warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CustomTkinter: tema e aspetto
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ---------------------------------------------------------------------------
# Registra i percorsi DLL NVIDIA installati via pip (cublas, cudnn, ecc.)
# ---------------------------------------------------------------------------
def _registra_nvidia_dll():
    """Aggiunge al PATH di sistema le cartelle lib/bin dei pacchetti nvidia-*
    installati via pip, così CTranslate2 può trovare cublas64_12.dll ecc."""
    try:
        import importlib.metadata
        nvidia_packages = [
            d.metadata["Name"] for d in importlib.metadata.distributions()
            if (d.metadata["Name"] or "").startswith("nvidia-")
        ]
        for pkg in nvidia_packages:
            try:
                files = importlib.metadata.files(pkg)
                if not files:
                    continue
                pkg_dir = str(files[0].locate().parent)
                # Risali alla radice del pacchetto
                for subdir in ("lib", "bin"):
                    candidate = os.path.join(pkg_dir, subdir)
                    if os.path.isdir(candidate) and candidate not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = candidate + os.pathsep + os.environ.get("PATH", "")
                # Prova anche le sottocartelle che contengono DLL
                for root, dirs, fnames in os.walk(pkg_dir):
                    for fname in fnames:
                        if fname.lower().endswith(".dll"):
                            if root not in os.environ.get("PATH", ""):
                                os.environ["PATH"] = root + os.pathsep + os.environ.get("PATH", "")
                            break
            except Exception:
                continue
    except Exception:
        pass

_registra_nvidia_dll()

# ---------------------------------------------------------------------------
# Check opzionale faster-whisper
# ---------------------------------------------------------------------------
WHISPER_DISPONIBILE = False
try:
    from faster_whisper import WhisperModel
    WHISPER_DISPONIBILE = True
except ImportError:
    pass

MODELLI_WHISPER = [
    "tiny", "base", "small", "medium",
    "large-v2", "large-v3", "large-v3-turbo",
]
ESTENSIONI_AUDIO = ('.wav', '.mp3', '.ogg', '.opus', '.m4a', '.flac', '.wma', '.aac')
ESTENSIONI_VIDEO = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm',
                    '.m4v', '.mpg', '.mpeg', '.ts', '.3gp')

# Colori custom
COLORE_ACCENT = "#1f6aa5"
COLORE_VERDE = "#2d8659"
COLORE_ROSSO = "#c0392b"
COLORE_GRIGIO = "#4a4a4a"
COLORE_SFONDO_LOG = "#1a1a2e"
COLORE_TESTO_LOG = "#e0e0e0"
COLORE_TESTO_DIM = "#888888"
COLORE_ARANCIO = "#d4781f"
COLORE_GIALLO = "#b8860b"
COLORE_VIOLA = "#7d3c98"

PROGRESSO_FILE = "_progresso.json"
BATCH_PROGRESSO_FILE = "_batch_progresso.json"


# ===========================================================================
# Utility cleanup CUDA — evita crash a fine trascrizione
# ===========================================================================
def _libera_risorse_cuda(model_ref, dev, callback=None):
    """Libera esplicitamente il modello Whisper e svuota la cache CUDA.

    Senza questo passo, il GC di Python può liberare il modello in modo
    non deterministico (anche dopo che la GUI ha mostrato il messagebox),
    causando un crash silenzioso del processo su alcune configurazioni CUDA.
    """
    try:
        if model_ref is not None:
            del model_ref
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass
    if dev == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
    if callback:
        callback("🧹 Risorse modello liberate")


# ===========================================================================
# FUNZIONI BACKEND (estrazione + splitting + trascrizione)
# ===========================================================================

def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def find_ffmpeg():
    possible_paths = [
        'ffmpeg.exe', 'ffmpeg',
        get_resource_path('ffmpeg.exe'),
        os.path.join(os.path.dirname(sys.executable), 'ffmpeg.exe'),
        os.path.join(os.path.dirname(sys.executable), 'ffmpeg'),
    ]
    for path in possible_paths:
        try:
            result = subprocess.run([path, '-version'],
                                    capture_output=True, timeout=5)
            if result.returncode == 0:
                return path
        except:
            continue
    return None


def setup_moviepy():
    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path:
        os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
        return True
    return False


def get_file_size_mb(file_path):
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except:
        return 0


def ordine_naturale(nome):
    """Ordinamento naturale basato sull'ultimo numero presente nel nome."""
    numeri = re.findall(r'\d+', nome)
    return int(numeri[-1]) if numeri else 0


def split_audio_with_ffmpeg(input_path, output_dir, target_size_mb, base_name,
                            callback=None, stop_event=None):
    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        raise Exception("FFmpeg non trovato")

    info_cmd = [ffmpeg_path, '-i', input_path, '-hide_banner']
    try:
        result = subprocess.run(info_cmd, capture_output=True, text=True, timeout=30)
        output = result.stderr
        duration_line = None
        for line in output.split('\n'):
            if 'Duration:' in line:
                duration_line = line
                break
        if not duration_line:
            raise Exception("Impossibile determinare la durata del file audio")
        duration_part = duration_line.split('Duration:')[1].split(',')[0].strip()
        time_parts = duration_part.split(':')
        total_seconds = (float(time_parts[0]) * 3600 +
                         float(time_parts[1]) * 60 +
                         float(time_parts[2]))
    except Exception as e:
        raise Exception(f"Errore info file: {e}")

    file_size_mb = get_file_size_mb(input_path)
    mb_per_second = file_size_mb / total_seconds if total_seconds > 0 else 0
    segment_duration = (target_size_mb / mb_per_second) if mb_per_second > 0 else 600
    num_segments = math.ceil(total_seconds / segment_duration)

    segments_created = []
    for i in range(num_segments):
        if stop_event and stop_event.is_set():
            if callback:
                callback(f"\n⏹  Splitting interrotto dall'utente al segmento {i+1}/{num_segments}")
            break

        start_time = i * segment_duration
        if start_time >= total_seconds:
            break
        segment_filename = f"{base_name}_segmento_{i + 1}.wav"
        segment_path = os.path.join(output_dir, segment_filename)
        cmd = [
            ffmpeg_path, '-i', input_path,
            '-ss', str(start_time),
            '-t', str(min(segment_duration, total_seconds - start_time)),
            '-acodec', 'pcm_s16le', '-y', segment_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode != 0:
                if callback:
                    callback(f"  ✗ Errore FFmpeg segmento {i+1}")
                continue
            segments_created.append(segment_path)
            if callback:
                callback(f"  ✓ Segmento {len(segments_created)}/{num_segments}: "
                         f"{segment_filename}")
        except subprocess.TimeoutExpired:
            if callback:
                callback(f"  ✗ Timeout segmento {i+1}")
        except Exception as e:
            if callback:
                callback(f"  ✗ Errore segmento {i+1}: {e}")

    return segments_created, segment_duration / 60


def estrai_e_splitta_video(video_path, output_dir, target_size_mb,
                           callback=None, stop_event=None):
    """Estrae l'audio da un video e lo segmenta in `output_dir`.

    Backend GUI-agnostico: usa solo `callback` per il log e `stop_event`
    per l'interruzione. È condiviso dal flusso singolo video e dal batch.

    Ritorna: (lista_segmenti, durata_media_min, tempo_splitting_secondi).
    Solleva Exception se il video non contiene audio.
    """
    from moviepy.video.io.VideoFileClip import VideoFileClip
    import contextlib, io

    t0 = time.time()
    if callback:
        callback("⏳ Caricamento video...")
    video = VideoFileClip(video_path)
    if video.audio is None:
        video.close()
        raise Exception("Il video non contiene audio.")

    if callback:
        callback("⏳ Estrazione audio...")
    audio = video.audio
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        temp_audio_path = tmp.name

    try:
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
    except Exception:
        audio.write_audiofile(temp_audio_path)

    if stop_event and stop_event.is_set():
        audio.close()
        video.close()
        try:
            os.unlink(temp_audio_path)
        except:
            pass
        if callback:
            callback("⏹  Interrotto dopo estrazione audio")
        return [], 0.0, time.time() - t0

    if callback:
        callback("⏳ Segmentazione audio...\n")

    segments, seg_dur_min = split_audio_with_ffmpeg(
        temp_audio_path, output_dir, target_size_mb, base_name,
        callback=callback, stop_event=stop_event)

    audio.close()
    video.close()
    try:
        os.unlink(temp_audio_path)
    except:
        pass

    return segments, seg_dur_min, time.time() - t0


def trova_file_audio(cartella):
    file_audio = [
        f for f in os.listdir(cartella)
        if os.path.isfile(os.path.join(cartella, f))
        and os.path.splitext(f)[1].lower() in ESTENSIONI_AUDIO
    ]
    file_audio.sort(key=ordine_naturale)
    return file_audio


def trova_file_video(cartella):
    """Ritorna i soli file video presenti direttamente nella cartella
    (non ricorsivo), ordinati in modo naturale."""
    file_video = [
        f for f in os.listdir(cartella)
        if os.path.isfile(os.path.join(cartella, f))
        and os.path.splitext(f)[1].lower() in ESTENSIONI_VIDEO
    ]
    file_video.sort(key=ordine_naturale)
    return file_video


# ---------------------------------------------------------------------------
# Checkpoint: salvataggio / caricamento progresso (per cartella di segmenti)
# ---------------------------------------------------------------------------

def _carica_progresso(cartella):
    path = os.path.join(cartella, PROGRESSO_FILE)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception:
            pass
    return {"completati": [], "risultati": [], "risultati_dettaglio": [], "durata_totale": 0.0}


def _salva_progresso(cartella, progresso):
    path = os.path.join(cartella, PROGRESSO_FILE)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(progresso, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _rimuovi_progresso(cartella):
    path = os.path.join(cartella, PROGRESSO_FILE)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _nome_output(cartella):
    nome_cartella = os.path.basename(os.path.normpath(cartella))
    return f"{nome_cartella}_trascrizione.txt"


def _scrivi_file_trascrizione(cartella, risultati, risultati_dettaglio,
                               durata_totale, formato, tempo_trascrizione=0):
    """Scrive/sovrascrive il file di trascrizione con tutti i risultati disponibili."""
    if formato == "dettagliato":
        output = _formatta_dettagliato(risultati_dettaglio, durata_totale, tempo_trascrizione)
    else:
        output = "\n\n".join(r for r in risultati if r)

    nome_file_out = _nome_output(cartella)
    percorso_output = os.path.join(cartella, nome_file_out)
    with open(percorso_output, "w", encoding="utf-8") as f:
        f.write(output)
    return percorso_output


# ---------------------------------------------------------------------------
# Checkpoint di livello BATCH (per progetto Wiki)
# ---------------------------------------------------------------------------

def _carica_batch_progresso(cartella_progetto):
    path = os.path.join(cartella_progetto, BATCH_PROGRESSO_FILE)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"completati": []}


def _salva_batch_progresso(cartella_progetto, dati):
    path = os.path.join(cartella_progetto, BATCH_PROGRESSO_FILE)
    try:
        os.makedirs(cartella_progetto, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dati, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Trascrizione con checkpoint + stop/pausa + file parziale
# ---------------------------------------------------------------------------

def trascrivi_segmenti(cartella, modello_nome="large-v3-turbo", lingua=None,
                       device="auto", formato="txt", callback=None,
                       stop_event=None, pause_event=None,
                       video_size_mb=0, tempo_splitting=0,
                       model_precaricato=None):
    if not WHISPER_DISPONIBILE:
        raise ImportError("faster-whisper non installato.\nInstalla con: pip install faster-whisper")

    if device == "auto":
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            dev = "cpu"
    else:
        dev = device

    compute = "float16" if dev == "cuda" else "int8"

    # Se ci viene passato un modello già caricato (batch), lo riusiamo e NON
    # lo liberiamo qui: la gestione del ciclo di vita è del chiamante.
    model = model_precaricato
    model_di_proprieta = model is None

    t0 = time.time()
    try:
        if model is None:
            if callback:
                callback(f"⏳ Caricamento modello '{modello_nome}' su {dev} ({compute})...")
            try:
                model = WhisperModel(modello_nome, device=dev, compute_type=compute)
            except Exception as e:
                if dev == "cuda":
                    if callback:
                        callback(f"⚠ CUDA non disponibile ({e})")
                        callback(f"⏳ Fallback automatico su CPU (int8)...")
                    dev = "cpu"
                    compute = "int8"
                    model = WhisperModel(modello_nome, device=dev, compute_type=compute)
                else:
                    raise
            dt = time.time() - t0
            if callback:
                callback(f"✓ Modello caricato in {dt:.1f}s (device={dev})\n")

        file_audio = trova_file_audio(cartella)
        if not file_audio:
            if callback:
                callback("⚠ Nessun file audio trovato.")
            return "", 0.0

        # Carica progresso precedente
        progresso = _carica_progresso(cartella)
        completati = set(progresso.get("completati", []))
        risultati = list(progresso.get("risultati", []))
        risultati_dettaglio = list(progresso.get("risultati_dettaglio", []))
        durata_totale = progresso.get("durata_totale", 0.0)

        file_da_fare = [f for f in file_audio if f not in completati]
        n_gia_fatti = len(file_audio) - len(file_da_fare)

        if n_gia_fatti > 0 and callback:
            callback(f"🔄 Ripresa: {n_gia_fatti}/{len(file_audio)} segmenti gia' trascritti, "
                     f"riprendo da segmento {n_gia_fatti + 1}\n")

        if not file_da_fare:
            if callback:
                callback("✅ Tutti i segmenti sono gia' stati trascritti.")
            percorso_output = _finalizza_trascrizione(
                cartella, risultati, risultati_dettaglio,
                durata_totale, file_audio, formato, 0, callback,
                video_size_mb=video_size_mb, tempo_splitting=tempo_splitting)
            return percorso_output, 0.0

        if callback:
            callback(f"📂 Trovati {len(file_audio)} file audio totali, "
                     f"{len(file_da_fare)} da trascrivere\n")

        t_inizio = time.time()
        interrotto = False

        for i, nome_file in enumerate(file_da_fare, 1):
            # --- Check STOP ---
            if stop_event and stop_event.is_set():
                interrotto = True
                if callback:
                    callback(f"\n⏹  Trascrizione interrotta dall'utente dopo "
                             f"{n_gia_fatti + i - 1}/{len(file_audio)} segmenti")
                break

            # --- Check PAUSA ---
            if pause_event and pause_event.is_set():
                if callback:
                    callback(f"⏸  In pausa... (clicca ▶ Riprendi per continuare)")
                while pause_event.is_set():
                    if stop_event and stop_event.is_set():
                        interrotto = True
                        break
                    time.sleep(0.3)
                if interrotto:
                    if callback:
                        callback(f"\n⏹  Trascrizione interrotta durante la pausa")
                    break
                if callback:
                    callback(f"▶  Ripresa trascrizione...")

            percorso = os.path.join(cartella, nome_file)
            indice_globale = n_gia_fatti + i
            if callback:
                callback(f"[{indice_globale}/{len(file_audio)}] {nome_file}...")

            try:
                segmenti_gen, info = model.transcribe(
                    percorso, language=lingua, beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
                    condition_on_previous_text=True, no_speech_threshold=0.6,
                )
                segmenti_lista = []
                testi = []
                for seg in segmenti_gen:
                    t = seg.text.strip()
                    if t:
                        segmenti_lista.append({"inizio": seg.start, "fine": seg.end, "testo": t})
                        testi.append(t)

                testo_unito = " ".join(testi)
                durata_totale += info.duration
                risultati.append(testo_unito)
                risultati_dettaglio.append({
                    "file": nome_file, "lingua": info.language,
                    "durata": info.duration, "testo": testo_unito,
                    "segmenti": segmenti_lista,
                })
                completati.add(nome_file)

                # Salva checkpoint dopo ogni segmento
                _salva_progresso(cartella, {
                    "completati": list(completati),
                    "risultati": risultati,
                    "risultati_dettaglio": risultati_dettaglio,
                    "durata_totale": durata_totale,
                })

                stato = "✓" if testo_unito else "— silenzio"
                if callback:
                    callback(f"  {stato} | {info.language} | {info.duration:.1f}s")
            except Exception as e:
                risultati.append("")
                if callback:
                    callback(f"  ✗ ERRORE: {e}")

        tempo_trascrizione = time.time() - t_inizio

        if interrotto:
            # Genera file di trascrizione parziale
            percorso_parziale = _scrivi_file_trascrizione(
                cartella, risultati, risultati_dettaglio,
                durata_totale, formato, tempo_trascrizione)

            n_completati = len(completati)
            if callback:
                callback(f"\n{'─'*50}")
                callback(f"💾 PROGRESSO SALVATO + FILE PARZIALE")
                callback(f"{'─'*50}")
                callback(f"   Segmenti completati:  {n_completati}/{len(file_audio)}")
                callback(f"   Rimanenti:            {len(file_audio) - n_completati}")
                callback(f"   File parziale:        {percorso_parziale}")
                callback(f"   Riprendi con 'Trascrivi Segmenti' o 'Trascrivi da Cartella'")
                callback(f"   (alla ripresa il file verra' aggiornato con i nuovi segmenti)")
            return "", tempo_trascrizione

        # Trascrizione completata al 100%
        percorso_output = _finalizza_trascrizione(
            cartella, risultati, risultati_dettaglio,
            durata_totale, file_audio, formato, tempo_trascrizione, callback,
            video_size_mb=video_size_mb, tempo_splitting=tempo_splitting)

        return percorso_output, tempo_trascrizione

    finally:
        # Libera il modello solo se l'abbiamo creato noi in questa chiamata.
        # In modalità batch il modello è condiviso e viene liberato dal
        # chiamante a fine ciclo.
        if model_di_proprieta:
            _libera_risorse_cuda(model, dev, callback=callback)


def _finalizza_trascrizione(cartella, risultati, risultati_dettaglio,
                            durata_totale, file_audio, formato,
                            tempo_trascrizione, callback,
                            video_size_mb=0, tempo_splitting=0):
    """Scrive il file finale, pulisce audio e rimuove il checkpoint."""

    percorso_output = _scrivi_file_trascrizione(
        cartella, risultati, risultati_dettaglio,
        durata_totale, formato, tempo_trascrizione)

    n_ok = sum(1 for r in risultati if r)
    if callback:
        callback(f"\n{'─'*50}")
        callback(f"📊 REPORT TRASCRIZIONE")
        callback(f"{'─'*50}")
        callback(f"   Segmenti trascritti:  {n_ok}/{len(file_audio)}")
        callback(f"   Audio totale:         {durata_totale/60:.1f} min")
        callback(f"   Tempo trascrizione:   {formatta_tempo(tempo_trascrizione)}")
        if tempo_trascrizione > 0:
            callback(f"   Velocita':            {durata_totale/tempo_trascrizione:.1f}x tempo reale")
        if video_size_mb > 0 and (tempo_splitting + tempo_trascrizione) > 0:
            sec_per_mb = (tempo_splitting + tempo_trascrizione) / video_size_mb
            callback(f"   Performance E2E:      {sec_per_mb:.2f} sec/MB")
        callback(f"   Salvato in:           {percorso_output}")

    # Pulizia segmenti audio (solo se completato al 100%)
    if n_ok > 0:
        file_rimossi = 0
        for nome_file in file_audio:
            try:
                percorso_file = os.path.join(cartella, nome_file)
                if os.path.exists(percorso_file):
                    os.remove(percorso_file)
                    file_rimossi += 1
            except Exception as e:
                if callback:
                    callback(f"  ⚠ Impossibile eliminare {nome_file}: {e}")
        if callback:
            callback(f"\n🗑  Pulizia: {file_rimossi}/{len(file_audio)} file audio eliminati")

    # Rimuovi checkpoint
    _rimuovi_progresso(cartella)

    return percorso_output


def _formatta_dettagliato(risultati, durata_totale, tempo_elab):
    righe = []
    righe.append("=" * 70)
    righe.append("TRASCRIZIONE AUDIO")
    righe.append(f"Durata audio totale: {durata_totale/60:.1f} min")
    righe.append(f"Tempo elaborazione: {tempo_elab:.1f}s")
    if tempo_elab > 0:
        righe.append(f"Velocita': {durata_totale/tempo_elab:.1f}x tempo reale")
    righe.append("=" * 70)
    righe.append("")
    for r in risultati:
        righe.append(f"--- {r['file']} ({r['lingua']}, {r['durata']:.1f}s) ---")
        if not r["testo"]:
            righe.append("  [silenzio]")
        else:
            for seg in r.get("segmenti", []):
                m_i, s_i = divmod(int(seg["inizio"]), 60)
                m_f, s_f = divmod(int(seg["fine"]), 60)
                righe.append(f"  [{m_i:02d}:{s_i:02d} -> {m_f:02d}:{s_f:02d}] {seg['testo']}")
        righe.append("")
    righe.append("=" * 70)
    righe.append("TESTO COMPLETO")
    righe.append("=" * 70)
    righe.append("")
    righe.append("\n\n".join(r["testo"] for r in risultati if r["testo"]))
    return "\n".join(righe)


def formatta_tempo(secondi):
    if secondi < 60:
        return f"{secondi:.1f}s"
    elif secondi < 3600:
        m, s = int(secondi // 60), int(secondi % 60)
        return f"{m}m {s}s"
    else:
        h = int(secondi // 3600)
        m, s = int((secondi % 3600) // 60), int(secondi % 60)
        return f"{h}h {m}m {s}s"


# ===========================================================================
# LLM WIKI — generazione struttura a partire dalle trascrizioni
# ===========================================================================

def _slug(nome):
    s = re.sub(r'[^\w\-]+', '_', nome, flags=re.UNICODE).strip('_')
    return s or "sorgente"


def _scrivi_sorgente_wiki(raw_dir, indice, video_name, testo, n_segmenti):
    """Scrive una trascrizione come sorgente markdown immutabile in raw/.
    Ritorna il percorso del file creato."""
    base = os.path.splitext(video_name)[0]
    nome_md = f"{indice:02d}_{_slug(base)}.md"
    path = os.path.join(raw_dir, nome_md)
    oggi = datetime.now().strftime('%Y-%m-%d')

    if not (testo or "").strip():
        corpo_testo = "_(Nessun parlato rilevato nella trascrizione.)_"
    else:
        corpo_testo = testo.strip()

    frontmatter = (
        "---\n"
        f"source: {video_name}\n"
        "type: transcript\n"
        f"date: {oggi}\n"
        f"index: {indice:02d}\n"
        f"segments: {n_segmenti}\n"
        "tags: [trascrizione, sorgente]\n"
        "---\n\n"
    )
    corpo = f"# Trascrizione — {base}\n\n{corpo_testo}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(frontmatter + corpo)
    return path


CLAUDE_SCHEMA = """# Schema della Wiki — Istruzioni per l'agente LLM

Questa cartella è una **LLM Wiki**: una base di conoscenza in markdown che TU
(l'agente) costruisci e mantieni a partire dalle trascrizioni video presenti in
`raw/`. L'utente cura le fonti e fa le domande; tu fai tutto il lavoro di sintesi,
collegamento e manutenzione. La cartella va aperta come vault in Obsidian.

A differenza di un classico RAG, qui la conoscenza non viene riscoperta a ogni
domanda: viene **compilata una volta e tenuta aggiornata**. La wiki è un artefatto
persistente che si arricchisce con ogni nuova sorgente e con ogni domanda.

## Struttura
- `raw/` — fonti **immutabili**: una trascrizione per video. Non modificarle mai.
- `wiki/` — pagine generate da te: sintesi, pagine entità, pagine concetto, confronti.
- `wiki/index.md` — catalogo di tutte le pagine. Aggiornalo a ogni ingest.
- `wiki/log.md` — registro cronologico append-only (ingest, query, lint).
- `CLAUDE.md` / `AGENTS.md` — questo schema. Co-evolvilo con l'utente nel tempo.

## Convenzioni
- Tutto in markdown, in italiano.
- Collegamenti con i wikilink di Obsidian: `[[Nome Pagina]]`.
- Nomi file: `entita_<nome>.md`, `concetto_<nome>.md`, `sintesi_<tema>.md`,
  `sorgente_<slug>.md`.
- Ogni pagina inizia con frontmatter YAML (vedi sotto).
- Quando un dato nuovo contraddice uno vecchio, segnalalo con un blocco
  `> [!warning]` e cita entrambe le fonti.
- Cita sempre le fonti `raw/` da cui proviene un'affermazione.

## Operazioni

### Ingest (una fonte alla volta, consigliato)
1. Leggi la trascrizione in `raw/`.
2. Discuti con l'utente i punti chiave.
3. Crea una pagina di sintesi della fonte: `wiki/sorgente_<slug>.md`.
4. Crea/aggiorna le pagine entità e concetto collegate (persone, progetti,
   decisioni, strumenti, temi ricorrenti).
5. Aggiorna i cross-reference esistenti.
6. Aggiorna `wiki/index.md`.
7. Aggiungi una riga a `wiki/log.md`:
   `## [AAAA-MM-GG] ingest | <titolo sorgente>`.
   Una singola fonte può toccare 10-15 pagine.

### Query
1. Leggi prima `wiki/index.md` per individuare le pagine rilevanti.
2. Apri le pagine, sintetizza la risposta con citazioni alle fonti `raw/`.
3. Se la risposta è preziosa, archiviala come nuova pagina in `wiki/` e
   aggiorna l'index: le esplorazioni devono accumularsi, non perdersi in chat.

### Lint (manutenzione periodica)
Cerca: contraddizioni tra pagine, affermazioni superate da fonti più recenti,
pagine orfane (senza link in entrata), concetti citati ma senza pagina propria,
cross-reference mancanti, lacune colmabili con una ricerca. Proponi nuove domande
da investigare e nuove fonti da cercare.

## Formati

### Frontmatter pagina wiki
```yaml
---
title: <titolo>
type: entita | concetto | sintesi | sorgente
fonti: [01_video1, 02_video2]
aggiornato: AAAA-MM-GG
tags: [...]
---
```

### index.md
Organizzato per categoria: Sorgenti, Entità, Concetti, Sintesi.
Ogni voce: `- [[pagina]] — riassunto in una riga`.

### log.md
Append-only. Ogni voce inizia con `## [AAAA-MM-GG] <tipo> | <titolo>`,
così è interrogabile con `grep "^## \\[" wiki/log.md | tail -5`.

## Flusso tipico per l'utente
1. Apri questa cartella come vault in Obsidian.
2. Apri un agente (es. Claude Code) nella stessa cartella.
3. "Leggi CLAUDE.md e fai l'ingest delle sorgenti in raw/, una alla volta."
4. Naviga il risultato in Obsidian (graph view, link, index).
"""

README_WIKI = """# LLM Wiki — Guida rapida

Questa cartella è un progetto **LLM Wiki**: una base di conoscenza in markdown
costruita a partire dalle trascrizioni dei tuoi video, mantenuta da un agente LLM
(es. Claude Code) e navigata in Obsidian.

## Come usarla
1. Apri **questa cartella** come vault in Obsidian.
2. Apri un agente LLM (es. Claude Code) nella stessa cartella.
3. Chiedi all'agente di leggere `CLAUDE.md` e di fare l'**ingest** delle sorgenti
   in `raw/`, una alla volta.
4. L'agente costruisce le pagine in `wiki/` e mantiene `wiki/index.md` e
   `wiki/log.md`.
5. Fai domande: l'agente risponde dalla wiki e archivia le risposte utili.

## Struttura
- `raw/` — trascrizioni dei video (fonti immutabili).
- `wiki/` — pagine generate dall'agente (sintesi, entità, concetti).
- `CLAUDE.md` / `AGENTS.md` — istruzioni per l'agente (lo schema).
- `_segmenti/` — cartelle di lavoro per video (puoi ignorarle/eliminarle).
"""


def genera_struttura_wiki(cartella_progetto, cartella_video_origine, callback=None):
    """Crea/aggiorna lo scaffold della LLM Wiki nel progetto.

    Non sovrascrive CLAUDE.md, README, index.md se già presenti (la wiki è
    di proprietà dell'agente): aggiunge solo ciò che manca e logga il batch.
    """
    raw_dir = os.path.join(cartella_progetto, "raw")
    wiki_dir = os.path.join(cartella_progetto, "wiki")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(wiki_dir, exist_ok=True)
    oggi = datetime.now().strftime('%Y-%m-%d')

    sorgenti_md = sorted(
        [f for f in os.listdir(raw_dir) if f.lower().endswith(".md")],
        key=ordine_naturale)

    # CLAUDE.md + AGENTS.md (solo se mancanti)
    claude_path = os.path.join(cartella_progetto, "CLAUDE.md")
    if not os.path.exists(claude_path):
        with open(claude_path, "w", encoding="utf-8") as f:
            f.write(CLAUDE_SCHEMA)
        try:
            with open(os.path.join(cartella_progetto, "AGENTS.md"),
                      "w", encoding="utf-8") as f:
                f.write(CLAUDE_SCHEMA)
        except Exception:
            pass

    # README_WIKI.md (solo se mancante)
    readme_path = os.path.join(cartella_progetto, "README_WIKI.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(README_WIKI)

    # wiki/index.md starter (solo se mancante)
    index_path = os.path.join(wiki_dir, "index.md")
    if not os.path.exists(index_path):
        righe = [
            "# Indice della Wiki", "",
            f"_Generato il {oggi}. Da qui in avanti lo mantiene l'agente LLM._", "",
            "## Sorgenti (raw)", "",
        ]
        for s in sorgenti_md:
            nome = os.path.splitext(s)[0]
            righe.append(f"- [[{nome}]] — _(da sintetizzare)_")
        righe += ["", "## Entità", "", "_(vuoto)_", "",
                  "## Concetti", "", "_(vuoto)_", "",
                  "## Sintesi", "", "_(vuoto)_", ""]
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("\n".join(righe))

    # wiki/log.md: crea se manca, poi append della voce di batch
    log_path = os.path.join(wiki_dir, "log.md")
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("# Log della Wiki\n\nRegistro cronologico append-only.\n\n")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"## [{oggi}] batch | preparate {len(sorgenti_md)} sorgenti "
                f"da '{os.path.basename(os.path.normpath(cartella_video_origine))}'\n")
        f.write("- Sorgenti aggiornate in `raw/` e pronte per l'ingest.\n")
        f.write("- Prossimo passo: l'agente fa l'ingest una fonte alla volta.\n\n")

    if callback:
        callback(f"📚 Struttura Wiki pronta/aggiornata in: {cartella_progetto}")


# ===========================================================================
# BATCH — processa tutti i video di una cartella e prepara la Wiki
# ===========================================================================

def elabora_cartella_video(cartella_video, target_size_mb, modello, lingua,
                           device, formato, genera_wiki=True,
                           callback=None, stop_event=None, pause_event=None):
    """Processa TUTTI i video di una cartella (estrai → splitta → trascrivi)
    e prepara la struttura LLM Wiki con tutte le trascrizioni come sorgenti.

    È resumable a livello di video (checkpoint di batch) e a livello di
    segmento (checkpoint per cartella). Carica il modello Whisper UNA sola
    volta e lo riusa per tutti i video.

    Ritorna un dict di riepilogo (o None se non ci sono video).
    """
    file_video = trova_file_video(cartella_video)
    if not file_video:
        if callback:
            callback("⚠ Nessun file video trovato nella cartella.")
        return None

    nome_progetto = os.path.basename(os.path.normpath(cartella_video))
    cartella_progetto = os.path.join(cartella_video, f"_Wiki_{nome_progetto}")
    raw_dir = os.path.join(cartella_progetto, "raw")
    seg_root = os.path.join(cartella_progetto, "_segmenti")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(seg_root, exist_ok=True)

    batch_prog = _carica_batch_progresso(cartella_progetto)
    completati = set(batch_prog.get("completati", []))

    if callback:
        callback(f"📁 Cartella: {cartella_video}")
        callback(f"🎞  Video trovati: {len(file_video)}")
        if completati:
            callback(f"🔄 Già completati in un run precedente: {len(completati)}")
        callback(f"📦 Progetto Wiki: {cartella_progetto}\n")

    # --- Risoluzione device + caricamento UNICO del modello ---
    if device == "auto":
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            dev = "cpu"
    else:
        dev = device
    compute = "float16" if dev == "cuda" else "int8"

    if callback:
        callback(f"⏳ Caricamento modello '{modello}' su {dev} ({compute}) (una sola volta)...")
    t_model = time.time()
    model = None
    try:
        model = WhisperModel(modello, device=dev, compute_type=compute)
    except Exception as e:
        if dev == "cuda":
            if callback:
                callback(f"⚠ CUDA non disponibile ({e}) — fallback su CPU (int8)")
            dev = "cpu"
            compute = "int8"
            model = WhisperModel(modello, device=dev, compute_type=compute)
        else:
            raise
    if callback:
        callback(f"✓ Modello caricato in {time.time()-t_model:.1f}s (device={dev})\n")

    t_batch0 = time.time()
    n_nuovi = 0

    try:
        for idx, video_name in enumerate(file_video, 1):
            if stop_event and stop_event.is_set():
                if callback:
                    callback(f"\n⏹  Batch interrotto prima del video {idx}/{len(file_video)}.")
                break

            if video_name in completati:
                if callback:
                    callback(f"⏭  [{idx}/{len(file_video)}] {video_name} già fatto — salto")
                continue

            if callback:
                callback(f"\n{'━'*50}")
                callback(f"🎬  [{idx}/{len(file_video)}] {video_name}")
                callback(f"{'━'*50}")

            video_path = os.path.join(cartella_video, video_name)
            base = os.path.splitext(video_name)[0]
            seg_dir = os.path.join(seg_root, f"{idx:02d}_{_slug(base)}")

            # --- Estrazione + splitting ---
            try:
                segments, seg_dur_min, t_split = estrai_e_splitta_video(
                    video_path, seg_dir, target_size_mb,
                    callback=callback, stop_event=stop_event)
            except Exception as e:
                if callback:
                    callback(f"  ✗ Errore su '{video_name}': {e} — passo al prossimo")
                continue

            if stop_event and stop_event.is_set():
                if callback:
                    callback("⏹  Batch interrotto durante lo splitting (progresso salvato).")
                break
            if not segments:
                if callback:
                    callback(f"  ⚠ Nessun segmento per '{video_name}' — salto")
                continue

            # --- Trascrizione (modello condiviso) ---
            video_mb = get_file_size_mb(video_path)
            percorso_txt, _ = trascrivi_segmenti(
                cartella=seg_dir, modello_nome=modello, lingua=lingua,
                device=dev, formato=formato, callback=callback,
                stop_event=stop_event, pause_event=pause_event,
                video_size_mb=video_mb, tempo_splitting=t_split,
                model_precaricato=model)

            if not percorso_txt:
                # Interrotto durante la trascrizione (progresso già salvato),
                # oppure nessun file: in entrambi i casi non marchiamo completato.
                if stop_event and stop_event.is_set():
                    if callback:
                        callback("⏹  Batch interrotto durante la trascrizione (progresso salvato).")
                    break
                continue

            # --- Scrivi la trascrizione come sorgente della Wiki ---
            try:
                with open(percorso_txt, "r", encoding="utf-8") as f:
                    testo = f.read()
            except Exception:
                testo = ""

            if genera_wiki:
                raw_md = _scrivi_sorgente_wiki(raw_dir, idx, video_name, testo, len(segments))
                if callback:
                    callback(f"  📝 Sorgente Wiki: {os.path.basename(raw_md)}")

            completati.add(video_name)
            n_nuovi += 1
            _salva_batch_progresso(cartella_progetto, {"completati": list(completati)})
            if callback:
                callback(f"  ✅ '{video_name}' completato")

    finally:
        # Libera il modello (condiviso) UNA volta sola, a fine batch.
        _libera_risorse_cuda(model, dev, callback=callback)

    # --- Genera/aggiorna struttura Wiki ---
    wiki_pronta = False
    if genera_wiki:
        raw_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(".md")] \
            if os.path.isdir(raw_dir) else []
        if raw_files:
            genera_struttura_wiki(cartella_progetto, cartella_video, callback=callback)
            wiki_pronta = True

    return {
        "cartella_progetto": cartella_progetto,
        "n_video": len(file_video),
        "n_completati": len(completati),
        "n_nuovi": n_nuovi,
        "tempo": time.time() - t_batch0,
        "wiki_pronta": wiki_pronta,
        "interrotto": bool(stop_event and stop_event.is_set()),
    }


# ===========================================================================
# GUI — CustomTkinter
# ===========================================================================

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Audio Splitter & Transcriber")
        self.geometry("700x920")
        self.minsize(650, 820)

        self.cartella_segmenti = None
        self.in_esecuzione = False
        self.video_size_mb = 0.0
        self.tempo_splitting = 0.0
        self.tempo_trascrizione = 0.0

        # Eventi per stop e pausa
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()

        # Grid principale: la riga del log si espande
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        self._crea_header()
        self._crea_sezione_video()
        self._crea_sezione_trascrizione()
        self._crea_sezione_batch()
        self._crea_sezione_log()
        self._crea_status_bar()

    # -------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------
    def _crea_header(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, padx=20, pady=(15, 5), sticky="ew")

        ctk.CTkLabel(
            header, text="🎬  Audio Splitter & Transcriber",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(side="left")

        self.tema_var = ctk.StringVar(value="dark")
        ctk.CTkSegmentedButton(
            header, values=["☀️ Light", "🌙 Dark"],
            command=self._cambia_tema,
            font=ctk.CTkFont(size=12),
            selected_color=COLORE_ACCENT,
        ).pack(side="right")

    def _cambia_tema(self, scelta):
        modo = "light" if "Light" in scelta else "dark"
        ctk.set_appearance_mode(modo)

    # -------------------------------------------------------------------
    # Sezione 1: Video → Splitting
    # -------------------------------------------------------------------
    def _crea_sezione_video(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=1, column=0, padx=20, pady=(5, 5), sticky="ew")
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame, text="1 │ Estrazione & Segmentazione (singolo video)",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, columnspan=3, padx=15, pady=(12, 8), sticky="w")

        ctk.CTkLabel(frame, text="Video:", anchor="w").grid(
            row=1, column=0, padx=(15, 5), pady=4, sticky="w")

        self.entry_video = ctk.CTkEntry(
            frame, placeholder_text="Seleziona un file video...",
            state="readonly", width=380,
        )
        self.entry_video.grid(row=1, column=1, padx=5, pady=4, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(
            frame, text="📁 Sfoglia", width=100,
            command=self._browse_file,
            fg_color=COLORE_GRIGIO, hover_color="#5a5a5a",
        ).grid(row=1, column=2, padx=(5, 15), pady=4)

        row_split = ctk.CTkFrame(frame, fg_color="transparent")
        row_split.grid(row=2, column=0, columnspan=3, padx=15, pady=(4, 12),
                       sticky="ew")

        ctk.CTkLabel(row_split, text="Max MB/segmento:").pack(
            side="left", padx=(0, 5))

        self.size_entry = ctk.CTkEntry(row_split, width=70, justify="center")
        self.size_entry.pack(side="left", padx=(0, 5))
        self.size_entry.insert(0, "3")

        ctk.CTkLabel(
            row_split, text="(es. 25 = ~25 MB)",
            text_color=COLORE_TESTO_DIM, font=ctk.CTkFont(size=11),
        ).pack(side="left", padx=(0, 15))

        self.btn_elabora_e_trascrivi = ctk.CTkButton(
            row_split, text="⚡ Elabora + Trascrivi",
            command=self._process_video_e_trascrivi,
            fg_color=COLORE_ARANCIO, hover_color="#b5651d",
            font=ctk.CTkFont(size=13, weight="bold"), height=36,
        )
        self.btn_elabora_e_trascrivi.pack(side="right", padx=(5, 0))

        self.btn_elabora = ctk.CTkButton(
            row_split, text="▶  Elabora Video",
            command=self._process_video,
            fg_color=COLORE_ACCENT, hover_color="#174f7a",
            font=ctk.CTkFont(size=13, weight="bold"), height=36,
        )
        self.btn_elabora.pack(side="right")

    # -------------------------------------------------------------------
    # Sezione 2: Trascrizione
    # -------------------------------------------------------------------
    def _crea_sezione_trascrizione(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=2, column=0, padx=20, pady=5, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="2 │ Trascrizione (faster-whisper)",
            font=ctk.CTkFont(size=14, weight="bold"), anchor="w",
        ).grid(row=0, column=0, columnspan=4, padx=15, pady=(12, 8), sticky="w")

        ctk.CTkLabel(frame, text="Modello:").grid(
            row=1, column=0, padx=(15, 5), pady=4, sticky="w")
        self.modello_var = ctk.StringVar(value="large-v3-turbo")
        ctk.CTkOptionMenu(
            frame, variable=self.modello_var, values=MODELLI_WHISPER, width=160,
        ).grid(row=1, column=1, padx=5, pady=4, sticky="w")

        ctk.CTkLabel(frame, text="Lingua:").grid(
            row=1, column=2, padx=(15, 5), pady=4, sticky="w")
        self.lingua_var = ctk.StringVar(value="Auto-detect")
        ctk.CTkOptionMenu(
            frame, variable=self.lingua_var, width=160,
            values=["Auto-detect", "it (Italiano)", "en (English)"],
        ).grid(row=1, column=3, padx=(5, 15), pady=4, sticky="w")

        ctk.CTkLabel(frame, text="Formato:").grid(
            row=2, column=0, padx=(15, 5), pady=4, sticky="w")
        self.formato_var = ctk.StringVar(value="txt")
        ctk.CTkOptionMenu(
            frame, variable=self.formato_var, values=["txt", "dettagliato"],
            width=160,
        ).grid(row=2, column=1, padx=5, pady=4, sticky="w")

        ctk.CTkLabel(frame, text="Device:").grid(
            row=2, column=2, padx=(15, 5), pady=4, sticky="w")
        self.device_var = ctk.StringVar(value="auto")
        ctk.CTkOptionMenu(
            frame, variable=self.device_var, values=["auto", "cpu", "cuda"],
            width=160,
        ).grid(row=2, column=3, padx=(5, 15), pady=4, sticky="w")

        row_btns = ctk.CTkFrame(frame, fg_color="transparent")
        row_btns.grid(row=3, column=0, columnspan=4, padx=15, pady=(8, 12),
                      sticky="ew")

        self.btn_trascrivi = ctk.CTkButton(
            row_btns, text="🎙  Trascrivi Segmenti",
            command=self._trascrivi_segmenti,
            fg_color=COLORE_VERDE, hover_color="#1e6b43",
            font=ctk.CTkFont(size=13, weight="bold"), height=36,
        )
        self.btn_trascrivi.pack(side="left", padx=(0, 10))

        self.btn_trascrivi_cartella = ctk.CTkButton(
            row_btns, text="📂  Trascrivi da Cartella...",
            command=self._trascrivi_da_cartella,
            fg_color=COLORE_VERDE, hover_color="#1e6b43",
            font=ctk.CTkFont(size=13, weight="bold"), height=36,
        )
        self.btn_trascrivi_cartella.pack(side="left")

        if not WHISPER_DISPONIBILE:
            self.btn_trascrivi.configure(state="disabled")
            self.btn_trascrivi_cartella.configure(state="disabled")
            self.btn_elabora_e_trascrivi.configure(state="disabled")
            ctk.CTkLabel(
                frame,
                text="⚠  faster-whisper non installato — pip install faster-whisper",
                text_color=COLORE_ROSSO, font=ctk.CTkFont(size=11),
            ).grid(row=4, column=0, columnspan=4, padx=15, pady=(0, 8), sticky="w")

    # -------------------------------------------------------------------
    # Sezione 3: Batch cartella → LLM Wiki
    # -------------------------------------------------------------------
    def _crea_sezione_batch(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=3, column=0, padx=20, pady=5, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="3 │ Batch Cartella → LLM Wiki",
            font=ctk.CTkFont(size=14, weight="bold"), anchor="w",
        ).grid(row=0, column=0, columnspan=3, padx=15, pady=(12, 8), sticky="w")

        ctk.CTkLabel(frame, text="Cartella video:", anchor="w").grid(
            row=1, column=0, padx=(15, 5), pady=4, sticky="w")

        self.entry_cartella_video = ctk.CTkEntry(
            frame, placeholder_text="Seleziona una cartella con più video...",
            state="readonly",
        )
        self.entry_cartella_video.grid(row=1, column=1, padx=5, pady=4, sticky="ew")

        ctk.CTkButton(
            frame, text="📁 Sfoglia", width=100,
            command=self._browse_cartella_video,
            fg_color=COLORE_GRIGIO, hover_color="#5a5a5a",
        ).grid(row=1, column=2, padx=(5, 15), pady=4)

        row_b = ctk.CTkFrame(frame, fg_color="transparent")
        row_b.grid(row=2, column=0, columnspan=3, padx=15, pady=(4, 12), sticky="ew")

        self.wiki_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            row_b, text="Genera struttura LLM Wiki (raw/ + wiki/ + CLAUDE.md)",
            variable=self.wiki_var, font=ctk.CTkFont(size=12),
        ).pack(side="left")

        self.btn_batch = ctk.CTkButton(
            row_b, text="🚀 Elabora Cartella + Wiki",
            command=self._elabora_cartella_batch,
            fg_color=COLORE_VIOLA, hover_color="#633974",
            font=ctk.CTkFont(size=13, weight="bold"), height=36,
        )
        self.btn_batch.pack(side="right")

        ctk.CTkLabel(
            frame,
            text="Usa modello/lingua/formato/device della sezione 2 e i MB/segmento "
                 "della sezione 1. Output: una cartella '_Wiki_<nome>' con tutte le trascrizioni.",
            text_color=COLORE_TESTO_DIM, font=ctk.CTkFont(size=11),
            anchor="w", justify="left", wraplength=620,
        ).grid(row=3, column=0, columnspan=3, padx=15, pady=(0, 10), sticky="w")

        if not WHISPER_DISPONIBILE:
            self.btn_batch.configure(state="disabled")

    # -------------------------------------------------------------------
    # Sezione 4: Log + Progress + Controlli Pausa/Stop
    # -------------------------------------------------------------------
    def _crea_sezione_log(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=4, column=0, padx=20, pady=5, sticky="nsew")
        frame.grid_rowconfigure(2, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        top_row = ctk.CTkFrame(frame, fg_color="transparent")
        top_row.grid(row=0, column=0, padx=15, pady=(10, 5), sticky="ew")
        top_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            top_row, text="Log",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        # Pulsanti Pausa e Stop
        self.ctrl_frame = ctk.CTkFrame(top_row, fg_color="transparent")
        self.ctrl_frame.grid(row=0, column=1, sticky="e")

        self.btn_pausa = ctk.CTkButton(
            self.ctrl_frame, text="⏸ Pausa", width=90,
            command=self._toggle_pausa,
            fg_color=COLORE_GIALLO, hover_color="#9a7209",
            font=ctk.CTkFont(size=12, weight="bold"), height=30,
        )

        self.btn_stop = ctk.CTkButton(
            self.ctrl_frame, text="⏹ Stop", width=90,
            command=self._richiedi_stop,
            fg_color=COLORE_ROSSO, hover_color="#a93226",
            font=ctk.CTkFont(size=12, weight="bold"), height=30,
        )

        self.progress = ctk.CTkProgressBar(frame, height=6)
        self.progress.grid(row=1, column=0, padx=15, pady=(0, 5), sticky="ew")
        self.progress.set(0)

        self.log_text = ctk.CTkTextbox(
            frame, font=ctk.CTkFont(family="Consolas", size=12),
            activate_scrollbars=True, wrap="word",
            fg_color=COLORE_SFONDO_LOG, text_color=COLORE_TESTO_LOG,
            corner_radius=8,
        )
        self.log_text.grid(row=2, column=0, padx=15, pady=(0, 12), sticky="nsew")

    # -------------------------------------------------------------------
    # Status bar
    # -------------------------------------------------------------------
    def _crea_status_bar(self):
        self.status_label = ctk.CTkLabel(
            self, text="  Pronto",
            font=ctk.CTkFont(size=11),
            text_color=COLORE_TESTO_DIM, anchor="w",
        )
        self.status_label.grid(row=5, column=0, padx=20, pady=(0, 8), sticky="ew")

    # ===================================================================
    # Utility
    # ===================================================================
    # -------------------------------------------------------------------
    # Logging su file
    # -------------------------------------------------------------------
    def _apri_log_file(self, etichetta="elaborazione"):
        """Apre un nuovo file di log nella cartella logs/ accanto a main.py."""
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            logs_dir = os.path.join(base_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_file = f"{ts}_{etichetta}.log"
            self._log_file = open(os.path.join(logs_dir, nome_file),
                                  "w", encoding="utf-8", buffering=1)
            intestazione = (
                f"{'='*60}\n"
                f"Log elaborazione — {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
                f"{'='*60}\n"
            )
            self._log_file.write(intestazione)
        except Exception as e:
            self._log_file = None
            print(f"[WARN] Impossibile aprire log file: {e}")

    def _chiudi_log_file(self):
        """Chiude il file di log corrente (se aperto)."""
        try:
            if getattr(self, "_log_file", None):
                self._log_file.write(
                    f"\n{'='*60}\n"
                    f"Fine — {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
                    f"{'='*60}\n"
                )
                self._log_file.close()
        except Exception:
            pass
        finally:
            self._log_file = None

    def _log(self, msg):
        # Scrivi sul file di log (se aperto)
        try:
            if getattr(self, "_log_file", None):
                ts = datetime.now().strftime("%H:%M:%S")
                self._log_file.write(f"[{ts}] {msg}\n")
        except Exception:
            pass
        # Aggiorna la GUI
        def _append():
            self.log_text.insert("end", msg + "\n")
            self.log_text.see("end")
        self.after(0, _append)

    def _set_status(self, msg):
        self.after(0, lambda: self.status_label.configure(text=f"  {msg}"))

    def _set_progress(self, valore):
        def _update():
            if valore < 0:
                self.progress.configure(mode="indeterminate")
                self.progress.start()
            else:
                self.progress.stop()
                self.progress.configure(mode="determinate")
                self.progress.set(valore)
        self.after(0, _update)

    def _mostra_controlli(self):
        self.btn_pausa.pack(side="left", padx=(0, 5))
        self.btn_stop.pack(side="left")

    def _nascondi_controlli(self):
        self.btn_pausa.pack_forget()
        self.btn_stop.pack_forget()

    def _toggle_pausa(self):
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.btn_pausa.configure(text="⏸ Pausa", fg_color=COLORE_GIALLO,
                                     hover_color="#9a7209")
            self._set_status("Ripresa in corso...")
        else:
            self.pause_event.set()
            self.btn_pausa.configure(text="▶ Riprendi", fg_color=COLORE_VERDE,
                                     hover_color="#1e6b43")
            self._set_status("In pausa")

    def _richiedi_stop(self):
        if messagebox.askyesno("Conferma Stop",
                "Vuoi interrompere l'esecuzione?\n\n"
                "Il progresso e la trascrizione parziale\n"
                "verranno salvati. Potrai riprendere\n"
                "in un secondo momento."):
            self.stop_event.set()
            self.pause_event.clear()
            self._set_status("Interruzione in corso...")
            self._log("⏹  Stop richiesto, attendo completamento segmento corrente...")

    def _set_in_esecuzione(self, stato):
        def _update():
            self.in_esecuzione = stato
            s = "disabled" if stato else "normal"
            self.btn_elabora.configure(state=s)
            self.btn_elabora_e_trascrivi.configure(
                state=s if WHISPER_DISPONIBILE else "disabled")
            wh_s = s if WHISPER_DISPONIBILE else "disabled"
            self.btn_trascrivi.configure(state=wh_s)
            self.btn_trascrivi_cartella.configure(state=wh_s)
            self.btn_batch.configure(state=wh_s)
            if stato:
                self._set_progress(-1)
                self.stop_event.clear()
                self.pause_event.clear()
                self.btn_pausa.configure(text="⏸ Pausa", fg_color=COLORE_GIALLO,
                                         hover_color="#9a7209")
                self._mostra_controlli()
            else:
                self._set_progress(0)
                self._nascondi_controlli()
        self.after(0, _update)

    def _stampa_report_e2e(self):
        tempo_e2e = self.tempo_splitting + self.tempo_trascrizione
        self._log(f"\n{'━'*50}")
        self._log(f"📋  REPORT FINALE END-TO-END")
        self._log(f"{'━'*50}")
        self._log(f"   Video sorgente:       {self.video_size_mb:.1f} MB")
        self._log(f"   Tempo splitting:      {formatta_tempo(self.tempo_splitting)}")
        self._log(f"   Tempo trascrizione:   {formatta_tempo(self.tempo_trascrizione)}")
        self._log(f"   Tempo totale E2E:     {formatta_tempo(tempo_e2e)}")
        if self.video_size_mb > 0 and tempo_e2e > 0:
            sec_per_mb = tempo_e2e / self.video_size_mb
            mb_per_min = (self.video_size_mb / tempo_e2e) * 60
            self._log(f"   Performance:          {sec_per_mb:.2f} sec/MB  ({mb_per_min:.1f} MB/min)")
        self._log(f"{'━'*50}")

    # ===================================================================
    # Selezione file / cartelle
    # ===================================================================
    def _browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Seleziona file video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
                ("All files", "*.*"),
            ]
        )
        if file_path:
            self.entry_video.configure(state="normal")
            self.entry_video.delete(0, "end")
            self.entry_video.insert(0, file_path)
            self.entry_video.configure(state="readonly")
            self._video_path = file_path

    def _browse_cartella_video(self):
        cartella = filedialog.askdirectory(title="Seleziona cartella con i video")
        if cartella:
            self.entry_cartella_video.configure(state="normal")
            self.entry_cartella_video.delete(0, "end")
            self.entry_cartella_video.insert(0, cartella)
            self.entry_cartella_video.configure(state="readonly")
            self._cartella_video_batch = cartella

    # ===================================================================
    # Elaborazione video (singolo)
    # ===================================================================
    def _validate_file_size(self):
        try:
            size_mb = float(self.size_entry.get().strip())
            if size_mb <= 0:
                messagebox.showerror("Errore", "La dimensione deve essere positiva.")
                return None
            if size_mb > 500:
                if not messagebox.askyesno("Conferma",
                        f"{size_mb} MB per segmento. Sicuro?"):
                    return None
            return size_mb
        except ValueError:
            messagebox.showerror("Errore", "Inserisci un numero valido.")
            return None

    def _process_video(self):
        self._avvia_elaborazione_video(auto_trascrivi=False)

    def _process_video_e_trascrivi(self):
        if not WHISPER_DISPONIBILE:
            messagebox.showerror("Errore",
                "faster-whisper non installato.\n"
                "Installa con: pip install faster-whisper")
            return
        self._avvia_elaborazione_video(auto_trascrivi=True)

    def _avvia_elaborazione_video(self, auto_trascrivi=False):
        if self.in_esecuzione:
            return
        if not setup_moviepy():
            messagebox.showerror("Errore",
                "FFmpeg non trovato.\n1. Installa FFmpeg nel PATH\n"
                "2. Oppure metti ffmpeg.exe nella cartella dell'eseguibile")
            return

        video_path = getattr(self, '_video_path', None)
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Errore", "Seleziona un file video valido.")
            return

        target_size_mb = self._validate_file_size()
        if target_size_mb is None:
            return

        self.video_size_mb = get_file_size_mb(video_path)
        self._set_in_esecuzione(True)

        # Apri log file
        etichetta = "elaborazione_trascrizione" if auto_trascrivi else "elaborazione_video"
        self._apri_log_file(etichetta)

        modalita = "ELABORAZIONE + TRASCRIZIONE" if auto_trascrivi else "ELABORAZIONE VIDEO"
        self._log(f"\n{'━'*50}")
        self._log(f"🎬  {modalita}")
        self._log(f"{'━'*50}")
        self._log(f"   File: {os.path.basename(video_path)}")
        self._log(f"   Dimensione: {self.video_size_mb:.1f} MB")
        if auto_trascrivi:
            self._log(f"   Modalita': one-click (split → trascrivi)\n")
        else:
            self._log("")

        lingua = self._get_lingua()
        modello = self.modello_var.get()
        formato = self.formato_var.get()
        device = self.device_var.get()

        def lavoro():
            try:
                self._set_status("Estrazione e segmentazione...")
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_dir = os.path.join(os.path.dirname(video_path), base_name)

                segments, seg_dur_min, t_split = estrai_e_splitta_video(
                    video_path, output_dir, target_size_mb,
                    callback=self._log, stop_event=self.stop_event)
                self.tempo_splitting = t_split

                if self.stop_event.is_set() and not segments:
                    self._set_status("Interrotto")
                    return

                if not segments:
                    self._set_status("Errore splitting")
                    self.after(0, lambda: messagebox.showerror("Errore",
                        "Nessun segmento creato."))
                    return

                self.cartella_segmenti = output_dir
                self._log(f"\n{'─'*50}")
                self._log(f"📊  REPORT SPLITTING")
                self._log(f"{'─'*50}")
                self._log(f"   Segmenti creati:   {len(segments)}")
                self._log(f"   Durata media:      ~{seg_dur_min:.1f} min/segmento")
                self._log(f"   Tempo splitting:   {formatta_tempo(self.tempo_splitting)}")
                self._log(f"   Cartella:          {output_dir}")

                if auto_trascrivi and not self.stop_event.is_set():
                    self._log(f"\n{'━'*50}")
                    self._log(f"🎙  TRASCRIZIONE AUTOMATICA")
                    self._log(f"{'━'*50}")
                    self._log(f"   Cartella: {output_dir}\n")
                    self._set_status("Trascrizione in corso...")

                    percorso_out, tempo_trasc = trascrivi_segmenti(
                        cartella=output_dir, modello_nome=modello,
                        lingua=lingua, device=device, formato=formato,
                        callback=self._log,
                        stop_event=self.stop_event,
                        pause_event=self.pause_event,
                        video_size_mb=self.video_size_mb,
                        tempo_splitting=self.tempo_splitting)

                    self.tempo_trascrizione = tempo_trasc

                    if percorso_out:
                        self._set_status("Elaborazione + Trascrizione completata!")
                        self._stampa_report_e2e()
                        self._log("\n🟢 App pronta — puoi consultare i log o avviare un'altra trascrizione.")
                        self.after(0, lambda: messagebox.showinfo("Completato",
                            f"Processo end-to-end completato!\n\n"
                            f"Segmenti: {len(segments)}\n"
                            f"Trascrizione: {percorso_out}\n"
                            f"Tempo totale: {formatta_tempo(self.tempo_splitting + self.tempo_trascrizione)}"))
                    else:
                        if self.stop_event.is_set():
                            self._set_status("Interrotto — progresso e file parziale salvati")
                        else:
                            self._set_status("Splitting OK, nessun file trascritto")
                elif self.stop_event.is_set():
                    self._set_status("Interrotto dopo splitting")
                    self._log(f"\n✅ Splitting completato. Trascrizione non avviata (interrotto).")
                    self._log(f"   Riprendi con 'Trascrivi Segmenti' o 'Trascrivi da Cartella'")
                else:
                    self._set_status("Splitting completato!")
                    self._log(f"\n✅ Puoi ora cliccare 'Trascrivi Segmenti'")
                    self.after(0, lambda: messagebox.showinfo("Successo",
                        f"{len(segments)} segmenti creati\n"
                        f"Tempo: {formatta_tempo(self.tempo_splitting)}\n"
                        f"Percorso: {output_dir}"))
            except ImportError as e:
                self._set_status("Errore trascrizione")
                self._log(f"\n✗ ERRORE: {e}")
                self.after(0, lambda: messagebox.showerror("Errore", str(e)))
            except Exception as e:
                self._set_status("Errore")
                err = str(e)
                if "'NoneType'" in err:
                    err = "FFmpeg non trovato o non configurato."
                self._log(f"✗ ERRORE: {err}")
                self.after(0, lambda: messagebox.showerror("Errore", err))
            finally:
                self._chiudi_log_file()
                self._set_in_esecuzione(False)

        threading.Thread(target=lavoro, daemon=True).start()

    # ===================================================================
    # Trascrizione (segmenti / cartella audio)
    # ===================================================================
    def _get_lingua(self):
        sel = self.lingua_var.get()
        if sel.startswith("it"):
            return "it"
        elif sel.startswith("en"):
            return "en"
        return None

    def _trascrivi_segmenti(self):
        if self.in_esecuzione:
            return
        if not self.cartella_segmenti or not os.path.isdir(self.cartella_segmenti):
            messagebox.showwarning("Attenzione",
                "Nessuna cartella disponibile.\n"
                "Elabora prima un video o usa 'Trascrivi da Cartella...'.")
            return
        self._esegui_trascrizione(self.cartella_segmenti)

    def _trascrivi_da_cartella(self):
        if self.in_esecuzione:
            return
        cartella = filedialog.askdirectory(title="Seleziona cartella con file audio")
        if cartella:
            self.cartella_segmenti = cartella
            self._esegui_trascrizione(cartella)

    def _esegui_trascrizione(self, cartella):
        self._set_in_esecuzione(True)
        # Apri log file (solo se non già aperto da _avvia_elaborazione_video)
        if not getattr(self, "_log_file", None):
            self._apri_log_file("trascrizione")
        self._log(f"\n{'━'*50}")
        self._log(f"🎙  TRASCRIZIONE")
        self._log(f"{'━'*50}")
        self._log(f"   Cartella: {cartella}\n")
        self._set_status("Trascrizione in corso...")

        lingua = self._get_lingua()
        modello = self.modello_var.get()
        formato = self.formato_var.get()
        device = self.device_var.get()

        def lavoro():
            try:
                percorso_out, tempo_trasc = trascrivi_segmenti(
                    cartella=cartella, modello_nome=modello,
                    lingua=lingua, device=device, formato=formato,
                    callback=self._log,
                    stop_event=self.stop_event,
                    pause_event=self.pause_event)

                self.tempo_trascrizione = tempo_trasc

                if percorso_out:
                    self._set_status("Trascrizione completata!")
                    if self.tempo_splitting > 0 and self.video_size_mb > 0:
                        self._stampa_report_e2e()
                    self._log("\n🟢 App pronta — puoi consultare i log o avviare un'altra trascrizione.")
                    self.after(0, lambda: messagebox.showinfo("Completato",
                        f"Trascrizione salvata in:\n{percorso_out}"))
                else:
                    if self.stop_event.is_set():
                        self._set_status("Interrotto — progresso e file parziale salvati")
                    else:
                        self._set_status("Nessun file trascritto")
            except Exception as e:
                self._set_status("Errore trascrizione")
                self._log(f"\n✗ ERRORE: {e}")
                self.after(0, lambda: messagebox.showerror("Errore", str(e)))
            finally:
                self._chiudi_log_file()
                self._set_in_esecuzione(False)

        threading.Thread(target=lavoro, daemon=True).start()

    # ===================================================================
    # Batch cartella → LLM Wiki
    # ===================================================================
    def _elabora_cartella_batch(self):
        if self.in_esecuzione:
            return
        if not WHISPER_DISPONIBILE:
            messagebox.showerror("Errore",
                "faster-whisper non installato.\n"
                "Installa con: pip install faster-whisper")
            return
        if not setup_moviepy():
            messagebox.showerror("Errore",
                "FFmpeg non trovato.\n1. Installa FFmpeg nel PATH\n"
                "2. Oppure metti ffmpeg.exe nella cartella dell'eseguibile")
            return

        cartella = getattr(self, "_cartella_video_batch", None)
        if not cartella or not os.path.isdir(cartella):
            messagebox.showerror("Errore", "Seleziona una cartella valida con i video.")
            return

        video = trova_file_video(cartella)
        if not video:
            messagebox.showwarning("Attenzione",
                "Nessun file video trovato nella cartella selezionata.")
            return

        target_size_mb = self._validate_file_size()
        if target_size_mb is None:
            return

        genera_wiki = bool(self.wiki_var.get())
        lingua = self._get_lingua()
        modello = self.modello_var.get()
        formato = self.formato_var.get()
        device = self.device_var.get()

        # Reset metriche E2E (non significative in batch multi-video)
        self.tempo_splitting = 0.0
        self.video_size_mb = 0.0

        self._set_in_esecuzione(True)
        self._apri_log_file("batch_wiki")
        self._log(f"\n{'━'*50}")
        self._log(f"🚀  BATCH CARTELLA → LLM WIKI")
        self._log(f"{'━'*50}")
        self._log(f"   Video da processare: {len(video)}")
        self._log(f"   Genera Wiki:         {'sì' if genera_wiki else 'no'}\n")
        self._set_status("Batch in corso...")

        def lavoro():
            try:
                risultato = elabora_cartella_video(
                    cartella_video=cartella, target_size_mb=target_size_mb,
                    modello=modello, lingua=lingua, device=device,
                    formato=formato, genera_wiki=genera_wiki,
                    callback=self._log,
                    stop_event=self.stop_event, pause_event=self.pause_event)

                if not risultato:
                    self._set_status("Nessun video elaborato")
                    return

                cp = risultato["cartella_progetto"]
                self._log(f"\n{'━'*50}")
                self._log(f"📋  REPORT BATCH")
                self._log(f"{'━'*50}")
                self._log(f"   Video totali:        {risultato['n_video']}")
                self._log(f"   Completati (tot):    {risultato['n_completati']}")
                self._log(f"   Nuovi in questo run: {risultato['n_nuovi']}")
                self._log(f"   Tempo batch:         {formatta_tempo(risultato['tempo'])}")
                self._log(f"   Progetto:            {cp}")
                self._log(f"{'━'*50}")

                if risultato["interrotto"]:
                    self._set_status("Batch interrotto — progresso salvato")
                    self._log("\n⏹  Interrotto. Riavvia il batch sulla stessa cartella per riprendere.")
                elif risultato["wiki_pronta"]:
                    self._set_status("Batch + Wiki completati!")
                    self._log(f"\n📚 LLM Wiki pronta.")
                    self._log(f"   1) Apri '{os.path.basename(cp)}' come vault in Obsidian")
                    self._log(f"   2) Apri un agente (es. Claude Code) nella stessa cartella")
                    self._log(f"   3) Chiedi: \"Leggi CLAUDE.md e fai l'ingest delle sorgenti in raw/\"")
                    self._log("\n🟢 App pronta.")
                    self.after(0, lambda: messagebox.showinfo("Completato",
                        f"Batch completato!\n\n"
                        f"Video completati: {risultato['n_completati']}/{risultato['n_video']}\n"
                        f"Progetto Wiki:\n{cp}\n\n"
                        f"Apri la cartella in Obsidian e usa un agente LLM "
                        f"(vedi CLAUDE.md) per costruire la wiki."))
                else:
                    self._set_status("Batch completato (senza Wiki)")
                    self.after(0, lambda: messagebox.showinfo("Completato",
                        f"Batch completato!\n\n"
                        f"Video completati: {risultato['n_completati']}/{risultato['n_video']}\n"
                        f"Trascrizioni nelle sottocartelle di:\n{cp}"))
            except Exception as e:
                self._set_status("Errore batch")
                self._log(f"\n✗ ERRORE BATCH: {e}")
                self.after(0, lambda: messagebox.showerror("Errore", str(e)))
            finally:
                self._chiudi_log_file()
                self._set_in_esecuzione(False)

        threading.Thread(target=lavoro, daemon=True).start()


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    app = App()
    app.mainloop()