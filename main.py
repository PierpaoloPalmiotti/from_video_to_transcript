import customtkinter as ctk
from tkinter import messagebox, filedialog
import os
import sys
import re
import math
import time
import warnings
import subprocess
import tempfile
import threading

# Sopprimi i warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CustomTkinter: tema e aspetto
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

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

# Colori custom
COLORE_ACCENT = "#1f6aa5"
COLORE_VERDE = "#2d8659"
COLORE_ROSSO = "#c0392b"
COLORE_GRIGIO = "#4a4a4a"
COLORE_SFONDO_LOG = "#1a1a2e"
COLORE_TESTO_LOG = "#e0e0e0"
COLORE_TESTO_DIM = "#888888"


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


def split_audio_with_ffmpeg(input_path, output_dir, target_size_mb, base_name,
                            callback=None):
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


def trova_file_audio(cartella):
    file_audio = [
        f for f in os.listdir(cartella)
        if os.path.isfile(os.path.join(cartella, f))
        and os.path.splitext(f)[1].lower() in ESTENSIONI_AUDIO
    ]
    def ordine_naturale(nome):
        numeri = re.findall(r'\d+', nome)
        return int(numeri[-1]) if numeri else 0
    file_audio.sort(key=ordine_naturale)
    return file_audio


def trascrivi_segmenti(cartella, modello_nome="large-v3-turbo", lingua=None,
                       device="auto", formato="txt", callback=None):
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
    if callback:
        callback(f"⏳ Caricamento modello '{modello_nome}' su {dev} ({compute})...")

    t0 = time.time()
    model = WhisperModel(modello_nome, device=dev, compute_type=compute)
    dt = time.time() - t0
    if callback:
        callback(f"✓ Modello caricato in {dt:.1f}s\n")

    file_audio = trova_file_audio(cartella)
    if not file_audio:
        if callback:
            callback("⚠ Nessun file audio trovato.")
        return "", 0.0

    if callback:
        callback(f"📂 Trovati {len(file_audio)} file audio\n")

    risultati = []
    risultati_dettaglio = []
    durata_totale = 0.0
    t_inizio = time.time()

    for i, nome_file in enumerate(file_audio, 1):
        percorso = os.path.join(cartella, nome_file)
        if callback:
            callback(f"[{i}/{len(file_audio)}] {nome_file}...")

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
            stato = "✓" if testo_unito else "— silenzio"
            if callback:
                callback(f"  {stato} | {info.language} | {info.duration:.1f}s")
        except Exception as e:
            risultati.append("")
            if callback:
                callback(f"  ✗ ERRORE: {e}")

    tempo_trascrizione = time.time() - t_inizio

    if formato == "dettagliato":
        output = _formatta_dettagliato(risultati_dettaglio, durata_totale, tempo_trascrizione)
    else:
        output = "\n\n".join(r for r in risultati if r)

    percorso_output = os.path.join(cartella, "trascrizione.txt")
    with open(percorso_output, "w", encoding="utf-8") as f:
        f.write(output)

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
        callback(f"   Salvato in:           {percorso_output}")

    return percorso_output, tempo_trascrizione


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
# GUI — CustomTkinter
# ===========================================================================

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Audio Splitter & Transcriber")
        self.geometry("700x780")
        self.minsize(650, 700)

        self.cartella_segmenti = None
        self.in_esecuzione = False
        self.video_size_mb = 0.0
        self.tempo_splitting = 0.0
        self.tempo_trascrizione = 0.0

        # Grid principale: la riga del log si espande
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        self._crea_header()
        self._crea_sezione_video()
        self._crea_sezione_trascrizione()
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

        # Toggle tema
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

        # Titolo sezione
        ctk.CTkLabel(
            frame, text="1 │ Estrazione & Segmentazione",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, columnspan=3, padx=15, pady=(12, 8), sticky="w")

        # Riga file
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

        # Riga dimensione + pulsante
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

        # Riga 1: Modello + Lingua
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

        # Riga 2: Formato + Device
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

        # Riga 3: Pulsanti
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

        # Avviso se manca faster-whisper
        if not WHISPER_DISPONIBILE:
            self.btn_trascrivi.configure(state="disabled")
            self.btn_trascrivi_cartella.configure(state="disabled")
            ctk.CTkLabel(
                frame,
                text="⚠  faster-whisper non installato — pip install faster-whisper",
                text_color=COLORE_ROSSO, font=ctk.CTkFont(size=11),
            ).grid(row=4, column=0, columnspan=4, padx=15, pady=(0, 8), sticky="w")

    # -------------------------------------------------------------------
    # Sezione 3: Log + Progress
    # -------------------------------------------------------------------
    def _crea_sezione_log(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=4, column=0, padx=20, pady=5, sticky="nsew")
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        # Riga titolo + progress bar
        top_row = ctk.CTkFrame(frame, fg_color="transparent")
        top_row.grid(row=0, column=0, padx=15, pady=(10, 5), sticky="ew")
        top_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            top_row, text="Log",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        self.progress = ctk.CTkProgressBar(top_row, height=6)
        self.progress.grid(row=0, column=1, padx=(15, 0), sticky="ew")
        self.progress.set(0)

        # Textbox log
        self.log_text = ctk.CTkTextbox(
            frame, font=ctk.CTkFont(family="Consolas", size=12),
            activate_scrollbars=True, wrap="word",
            fg_color=COLORE_SFONDO_LOG, text_color=COLORE_TESTO_LOG,
            corner_radius=8,
        )
        self.log_text.grid(row=1, column=0, padx=15, pady=(0, 12), sticky="nsew")

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
    def _log(self, msg):
        def _append():
            self.log_text.insert("end", msg + "\n")
            self.log_text.see("end")
        self.after(0, _append)

    def _set_status(self, msg):
        self.after(0, lambda: self.status_label.configure(text=f"  {msg}"))

    def _set_progress(self, valore):
        """valore da 0.0 a 1.0, oppure -1 per indeterminato."""
        def _update():
            if valore < 0:
                self.progress.configure(mode="indeterminate")
                self.progress.start()
            else:
                self.progress.stop()
                self.progress.configure(mode="determinate")
                self.progress.set(valore)
        self.after(0, _update)

    def _set_in_esecuzione(self, stato):
        def _update():
            self.in_esecuzione = stato
            s = "disabled" if stato else "normal"
            self.btn_elabora.configure(state=s)
            wh_s = s if WHISPER_DISPONIBILE else "disabled"
            self.btn_trascrivi.configure(state=wh_s)
            self.btn_trascrivi_cartella.configure(state=wh_s)
            if stato:
                self._set_progress(-1)
            else:
                self._set_progress(0)
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
    # Selezione file
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

    # ===================================================================
    # Elaborazione video
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
        self._log(f"\n{'━'*50}")
        self._log(f"🎬  ELABORAZIONE VIDEO")
        self._log(f"{'━'*50}")
        self._log(f"   File: {os.path.basename(video_path)}")
        self._log(f"   Dimensione: {self.video_size_mb:.1f} MB\n")

        def lavoro():
            try:
                from moviepy.video.io.VideoFileClip import VideoFileClip

                t_split_start = time.time()
                self._set_status("Caricamento video...")
                self._log("⏳ Caricamento video...")

                video = VideoFileClip(video_path)
                if video.audio is None:
                    self.after(0, lambda: messagebox.showerror(
                        "Errore", "Il video non contiene audio."))
                    video.close()
                    return

                self._set_status("Estrazione audio...")
                self._log("⏳ Estrazione audio...")

                audio = video.audio
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_base_dir = os.path.dirname(video_path)

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    temp_audio_path = tmp.name

                try:
                    import contextlib, io
                    f = io.StringIO()
                    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                        audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
                except Exception:
                    audio.write_audiofile(temp_audio_path)

                self._set_status("Segmentazione audio...")
                self._log("⏳ Segmentazione audio...\n")

                output_dir = os.path.join(output_base_dir, base_name)
                os.makedirs(output_dir, exist_ok=True)

                segments, seg_dur_min = split_audio_with_ffmpeg(
                    temp_audio_path, output_dir, target_size_mb, base_name,
                    callback=self._log)

                audio.close()
                video.close()
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass

                self.tempo_splitting = time.time() - t_split_start

                if segments:
                    self.cartella_segmenti = output_dir
                    self._set_status("Splitting completato!")
                    self._log(f"\n{'─'*50}")
                    self._log(f"📊  REPORT SPLITTING")
                    self._log(f"{'─'*50}")
                    self._log(f"   Segmenti creati:   {len(segments)}")
                    self._log(f"   Durata media:      ~{seg_dur_min:.1f} min/segmento")
                    self._log(f"   Tempo splitting:   {formatta_tempo(self.tempo_splitting)}")
                    self._log(f"   Cartella:          {output_dir}")
                    self._log(f"\n✅ Puoi ora cliccare 'Trascrivi Segmenti'")
                    self.after(0, lambda: messagebox.showinfo("Successo",
                        f"{len(segments)} segmenti creati\n"
                        f"Tempo: {formatta_tempo(self.tempo_splitting)}\n"
                        f"Percorso: {output_dir}"))
                else:
                    self._set_status("Errore splitting")
                    self.after(0, lambda: messagebox.showerror("Errore",
                        "Nessun segmento creato."))
            except Exception as e:
                self._set_status("Errore")
                err = str(e)
                if "'NoneType'" in err:
                    err = "FFmpeg non trovato o non configurato."
                self._log(f"✗ ERRORE: {err}")
                self.after(0, lambda: messagebox.showerror("Errore", err))
            finally:
                self._set_in_esecuzione(False)

        threading.Thread(target=lavoro, daemon=True).start()

    # ===================================================================
    # Trascrizione
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
                    callback=self._log)

                self.tempo_trascrizione = tempo_trasc

                if percorso_out:
                    self._set_status("Trascrizione completata!")
                    if self.tempo_splitting > 0 and self.video_size_mb > 0:
                        self._stampa_report_e2e()
                    self.after(0, lambda: messagebox.showinfo("Completato",
                        f"Trascrizione salvata in:\n{percorso_out}"))
                else:
                    self._set_status("Nessun file trascritto")
            except Exception as e:
                self._set_status("Errore trascrizione")
                self._log(f"\n✗ ERRORE: {e}")
                self.after(0, lambda: messagebox.showerror("Errore", str(e)))
            finally:
                self._set_in_esecuzione(False)

        threading.Thread(target=lavoro, daemon=True).start()


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    app = App()
    app.mainloop()