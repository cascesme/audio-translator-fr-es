#!/usr/bin/env python3
"""
French audio -> Spanish text (and optional Spanish speech)
- Input: a single audio/video file OR a directory of files
- ASR: faster-whisper (Whisper via CTranslate2)
- Translation: Argos Translate (FR -> EN -> ES) offline
- Optional TTS: Coqui TTS (Spanish)

Outputs per input file <name>:
  <out_prefix>_<name>.fr.txt   (French transcript)
  <out_prefix>_<name>.es.txt   (Spanish translation)
  <out_prefix>_<name>.es.wav   (Spanish audio, if TTS enabled)
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# ASR
from faster_whisper import WhisperModel

# Translation
import argostranslate.translate as argo

SUPPORTED_EXTS = {
    ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".oga", ".opus",
    ".aac", ".wma", ".mp4", ".mkv", ".webm"
}


def is_supported_audio(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in SUPPORTED_EXTS


def gather_inputs(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if not is_supported_audio(input_path):
            raise SystemExit(f"Unsupported format: {input_path.suffix} (needs one of {sorted(SUPPORTED_EXTS)})")
        return [input_path]
    if input_path.is_dir():
        files = sorted([p for p in input_path.iterdir() if is_supported_audio(p)])
        if not files:
            raise SystemExit(f"No audio files found in folder: {input_path}")
        return files
    raise SystemExit(f"Input not found: {input_path}")


def ensure_argos_translators_via_english(src_code: str, tgt_code: str):
    """
    Returns (src->en translator, en->tgt translator).
    Requires the Argos packages to be installed in the image already.
    """
    installed = argo.get_installed_languages()

    def get_lang(code: str):
        return next((l for l in installed if l.code == code), None)

    src = get_lang(src_code)
    en = get_lang("en")
    tgt = get_lang(tgt_code)

    missing = [c for c, lang in [(src_code, src), ("en", en), (tgt_code, tgt)] if lang is None]
    if missing:
        raise SystemExit(
            "Argos languages missing in the container: "
            f"{missing}. Make sure you baked the packages for "
            f"{src_code}->en and en->{tgt_code} into the image."
        )

    return src.get_translation(en), en.get_translation(tgt)


def init_tts_or_none(enable_tts: bool):
    if not enable_tts:
        return None
    try:
        from TTS.api import TTS
        # Use cached/baked model
        return TTS(model_name="tts_models/es/css10/vits")
    except Exception as e:
        print(f"Spanish TTS initialization failed (continuing without TTS): {e}", file=sys.stderr)
        return None


def transcribe_file(model: WhisperModel, path: Path, src_lang: str, beam_size: int) -> str:
    print(f"Transcribing: {path.name}", file=sys.stderr)
    segments, info = model.transcribe(
        str(path),
        language=src_lang,
        task="transcribe",
        beam_size=beam_size,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    # Stitch segments into a single line; you can change this to keep timestamps if you want
    return " ".join(seg.text.strip() for seg in segments).strip()


def translate_text(fr_text: str, fr_en, en_es) -> str:
    en_text = fr_en.translate(fr_text)
    es_text = en_es.translate(en_text)
    return es_text


def maybe_synthesize_tts(tts_api, text: str, out_wav: Path):
    if tts_api is None:
        return
    print(f"Synthesizing Spanish audio: {out_wav.name}", file=sys.stderr)
    # Save WAV (defaults are fine)
    tts_api.tts_to_file(text=text, file_path=str(out_wav), speaker=None, language=None)


def main():
    parser = argparse.ArgumentParser(
        description="French audio -> Spanish text (and optional Spanish speech)"
    )
    parser.add_argument("input", help="Audio/video file OR folder of files")
    parser.add_argument("--src", default="fr", help="Source language code (default: fr)")
    parser.add_argument("--tgt", default="es", help="Target language code (default: es)")
    parser.add_argument("--model-size", dest="model_size", default="small",
                        help="Whisper size: tiny|base|small|medium|large-v3 (default: small)")
    parser.add_argument("--beam-size", type=int, default=5, help="ASR beam size (default: 5)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Inference device (default: cpu)")
    parser.add_argument("--compute-type", default="int8",
                        help="CTranslate2 compute type: int8, int8_float16, float16, float32")
    parser.add_argument("--no-tts", action="store_true", help="Disable Spanish TTS output")
    parser.add_argument("--out-prefix", default="/data/output",
                        help="Prefix for output files (default: /data/output)")
    args = parser.parse_args()

    input_path = Path(args.input)
    files = gather_inputs(input_path)

    # Prepare translators (offline; expects fr->en and en->es installed)
    print("Preparing translators FR -> EN -> ES…", file=sys.stderr)
    fr_en, en_es = ensure_argos_translators_via_english(args.src, args.tgt)

    # Load Whisper once (big speedup for many files)
    print(f"Loading Whisper model ({args.model_size}, device={args.device}, compute_type={args.compute_type})…", file=sys.stderr)
    model = WhisperModel(args.model_size, device=args.device, compute_type=args.compute_type)

    # Optional TTS (loaded once)
    tts_api = init_tts_or_none(not args.no_tts)

    # Process each file
    for idx, f in enumerate(files, 1):
        print(f"\n--- [{idx}/{len(files)}] Processing {f.name} ---", file=sys.stderr)
        fr_text = transcribe_file(model, f, args.src, args.beam_size)

        # Write French transcript
        out_fr = Path(f"{args.out_prefix}/{f.stem}.fr.txt")
        out_fr.parent.mkdir(parents=True, exist_ok=True)
        out_fr.write_text(fr_text + "\n", encoding="utf-8")
        print(f"Wrote French transcript: {out_fr}", file=sys.stderr)

        # Translate via English pivot
        print("Translating FR -> EN -> ES…", file=sys.stderr)
        es_text = translate_text(fr_text, fr_en, en_es)

        # Write Spanish text
        out_es = Path(f"{args.out_prefix}/{f.stem}.es.txt")
        out_es.write_text(es_text + "\n", encoding="utf-8")
        print(f"Wrote Spanish translation: {out_es}", file=sys.stderr)

        # Optional Spanish speech
        if tts_api is not None:
            out_wav = Path(f"{args.out_prefix}_{f.stem}.es.wav")
            try:
                maybe_synthesize_tts(tts_api, es_text, out_wav)
                print(f"Wrote Spanish audio: {out_wav}", file=sys.stderr)
            except Exception as e:
                print(f"TTS failed for {f.name} (continuing): {e}", file=sys.stderr)

    print("\nAll done.", file=sys.stderr)


if __name__ == "__main__":
    main()