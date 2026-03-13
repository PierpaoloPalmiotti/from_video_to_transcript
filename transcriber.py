"""
transcriber.py — Trascrizione locale di segmenti audio con faster-whisper
Modulo standalone. Funziona indipendentemente dalla GUI.

Uso da riga di comando:
    python transcriber.py segmenti_audio/
    python transcriber.py segmenti_audio/ --lingua it
    python transcriber.py segmenti_audio/ --modello medium --formato dettagliato
    python transcriber.py segmenti_audio/ --output risultato.txt --device cuda

Uso come import:
    from transcriber import trascrivi_segmenti
    percorso = trascrivi_segmenti("segmenti_audio/", callback=print)
"""

import os
import re
import sys
import time
import argparse

ESTENSIONI_AUDIO = ('.wav', '.mp3', '.ogg', '.opus', '.m4a', '.flac', '.wma', '.aac')

MODELLI_DISPONIBILI = [
    "tiny", "base", "small", "medium",
    "large-v2", "large-v3", "large-v3-turbo",
]


def trova_file_audio(cartella):
    """Trova e ordina file audio in ordine naturale."""
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
                       device="auto", formato="txt", output_file=None,
                       beam_size=5, no_vad=False, callback=None):
    """
    Trascrive tutti i file audio in una cartella e salva un file unificato.

    Args:
        cartella:      percorso alla cartella con i file audio
        modello_nome:  modello Whisper (es. "large-v3-turbo", "medium")
        lingua:        "it", "en" o None per auto-detect
        device:        "cpu", "cuda" o "auto"
        formato:       "txt" (solo testo) o "dettagliato" (con timestamp)
        output_file:   percorso file output (None = trascrizione.txt nella cartella)
        beam_size:     beam search size (5 = qualita', 1 = veloce)
        no_vad:        True per disabilitare il filtro VAD
        callback:      funzione callback(messaggio) per log

    Returns:
        Percorso del file di trascrizione generato, o "" se nessun file trovato.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "faster-whisper non installato.\n"
            "Installa con: pip install faster-whisper"
        )

    if not os.path.isdir(cartella):
        raise FileNotFoundError(f"Cartella non trovata: {cartella}")

    # Auto-detect device
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
        callback(f"Caricamento modello '{modello_nome}' su {dev} ({compute})...")

    t0 = time.time()
    model = WhisperModel(modello_nome, device=dev, compute_type=compute)
    dt = time.time() - t0

    if callback:
        callback(f"Modello caricato in {dt:.1f}s\n")

    # Trova file audio
    file_audio = trova_file_audio(cartella)
    if not file_audio:
        if callback:
            callback("Nessun file audio trovato nella cartella.")
        return ""

    if callback:
        callback(f"Trovati {len(file_audio)} file audio da trascrivere.\n")

    # Trascrivi ogni file
    risultati = []
    risultati_dettaglio = []
    durata_totale = 0.0
    t_inizio = time.time()

    for i, nome_file in enumerate(file_audio, 1):
        percorso = os.path.join(cartella, nome_file)

        if callback:
            callback(f"[{i}/{len(file_audio)}] {nome_file}...")

        try:
            params = dict(
                language=lingua,
                beam_size=beam_size,
                vad_filter=not no_vad,
                condition_on_previous_text=True,
                no_speech_threshold=0.6,
            )
            if not no_vad:
                params["vad_parameters"] = dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200,
                )

            segmenti_gen, info = model.transcribe(percorso, **params)

            segmenti_lista = []
            testi = []
            for seg in segmenti_gen:
                t = seg.text.strip()
                if t:
                    segmenti_lista.append({
                        "inizio": seg.start,
                        "fine": seg.end,
                        "testo": t,
                    })
                    testi.append(t)

            testo_unito = " ".join(testi)
            durata_totale += info.duration

            risultati.append(testo_unito)
            risultati_dettaglio.append({
                "file": nome_file,
                "lingua": info.language,
                "durata": info.duration,
                "testo": testo_unito,
                "segmenti": segmenti_lista,
            })

            stato = "OK" if testo_unito else "(silenzio)"
            if callback:
                callback(f"  -> {stato} | {info.language} | {info.duration:.1f}s")

        except Exception as e:
            risultati.append("")
            if callback:
                callback(f"  -> ERRORE: {e}")

    tempo_totale = time.time() - t_inizio

    # Genera output
    if formato == "dettagliato":
        output = _formatta_dettagliato(risultati_dettaglio, durata_totale,
                                       tempo_totale)
    else:
        output = "\n\n".join(r for r in risultati if r)

    # Salva
    if output_file is None:
        output_file = os.path.join(cartella, "trascrizione.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output)

    n_ok = sum(1 for r in risultati if r)
    if callback:
        callback(f"\n{'='*50}")
        callback(f"Trascrizione completata in {tempo_totale:.1f}s")
        callback(f"Segmenti trascritti: {n_ok}/{len(file_audio)}")
        callback(f"Audio totale: {durata_totale/60:.1f} min")
        if tempo_totale > 0:
            callback(f"Velocita': {durata_totale/tempo_totale:.1f}x tempo reale")
        callback(f"Salvato in: {output_file}")

    return output_file


def _formatta_dettagliato(risultati, durata_totale, tempo_elab):
    """Formato dettagliato con timestamp per ogni frase."""
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
                righe.append(
                    f"  [{m_i:02d}:{s_i:02d} -> {m_f:02d}:{s_f:02d}] {seg['testo']}"
                )
        righe.append("")

    righe.append("=" * 70)
    righe.append("TESTO COMPLETO")
    righe.append("=" * 70)
    righe.append("")
    righe.append("\n\n".join(r["testo"] for r in risultati if r["testo"]))

    return "\n".join(righe)


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Trascrivi segmenti audio con faster-whisper (locale)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  python transcriber.py segmenti_audio/
  python transcriber.py segmenti_audio/ --lingua it
  python transcriber.py segmenti_audio/ --modello medium --formato dettagliato
  python transcriber.py segmenti_audio/ --output mia_trascrizione.txt
        """
    )
    parser.add_argument("cartella",
                        help="Cartella contenente i file audio")
    parser.add_argument("--modello", default="large-v3-turbo",
                        choices=MODELLI_DISPONIBILI,
                        help="Modello Whisper (default: large-v3-turbo)")
    parser.add_argument("--lingua", default=None,
                        help="Codice lingua: 'it', 'en', ecc. "
                             "(default: auto-detect)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device (default: auto)")
    parser.add_argument("--beam-size", type=int, default=5,
                        help="Beam size (default: 5)")
    parser.add_argument("--formato", default="txt",
                        choices=["txt", "dettagliato"],
                        help="Formato output (default: txt)")
    parser.add_argument("--output", default=None,
                        help="Percorso file output (default: trascrizione.txt)")
    parser.add_argument("--no-vad", action="store_true",
                        help="Disabilita filtro VAD (non consigliato)")

    args = parser.parse_args()

    trascrivi_segmenti(
        cartella=args.cartella,
        modello_nome=args.modello,
        lingua=args.lingua,
        device=args.device,
        beam_size=args.beam_size,
        formato=args.formato,
        output_file=args.output,
        no_vad=args.no_vad,
        callback=print,
    )


if __name__ == "__main__":
    main()