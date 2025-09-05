#!/usr/bin/env python3
"""
French audio -> Spanish text (and optional Spanish speech)

Features:
- Input: single audio/video file OR a directory of files
- Formats: anything ffmpeg decodes (.wav, .mp3, .ogg, .m4a, .flac, .mp4, .mkv, .webm, ...)
- ASR: faster-whisper (Whisper via CTranslate2)
- Translation: Argos Translate (FR -> EN -> ES) using preinstalled models
  (bake fr->en and en->es .argosmodel packages into the image)
- TTS: Coqui TTS (Spanish) -> default WAV then optional re-encode to mp3/ogg/opus/m4a
- Skips TTS on empty text; skips whole file if no speech detected
- Cleans up .txt artifacts after each file (keep only final audio if TTS enabled)

Outputs per input <name>:
  <out_prefix>_<name>.es.<ext>  (Spanish speech; ext set by --audio-format)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

from faster_whisper import WhisperModel
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
            raise SystemExit(f"Unsupported format: {input_path.suffix}")
        return [input_path]
    if input_path.is_dir():
        files = sorted([p for p in input_path.iterdir() if is_supported_audio(p)])
        if not files:
            raise SystemExit(f"No audio files found in {input_path}")
        return files
    raise SystemExit(f"Input not found: {input_path}")


def ensure_argos_translators_via_english(src_code: str, tgt_code: str):
    """Return (src->en translator, en->tgt translator); requires models already installed."""
    installed = argo.get_installed_languages()

    def get_lang(code: str):
        return next((l for l in installed if l.code == code), None)

    src = get_lang(src_code)
    en = get_lang("en")
    tgt = get_lang(tgt_code)

    missing = [c for c, lang in [(src_code, src), ("en", en), (tgt_code, tgt)] if lang is None]
    if missing:
        raise SystemExit(
            "Missing Argos languages in the container: "
            f"{missing}. Ensure you baked fr->en and en->{tgt_code} packages into the image."
        )

    return src.get_translation(en), en.get_translation(tgt)


def init_tts_or_none(enable_tts: bool):
    if not enable_tts:
        return None
    try:
        from TTS.api import TTS
        # Use a Spanish model that you've cached/baked into the image
        return TTS(model_name="tts_models/es/css10/vits")
    except Exception as e:
        print(f"Spanish TTS init failed (continuing without TTS): {e}", file=sys.stderr)
        return None


def transcribe_file(model: WhisperModel, path: Path, src_lang: str, beam_size: int) -> str:
    print(f"Transcribing {path.name}", file=sys.stderr)
    segments, _ = model.transcribe(
        str(path),
        language=src_lang,
        task="transcribe",
        beam_size=beam_size,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text


def translate_text(fr_text: str, fr_en, en_es) -> str:
    en_text = fr_en.translate(fr_text)
    es_text = en_es.translate(en_text)
    return es_text


def maybe_synthesize_tts(tts_api, text: str, out_base: Path, fmt: str):
    """
    Synthesize to WAV first (what TTS expects), then re-encode with ffmpeg if needed.
    Returns the final audio Path or None.
    """
    if tts_api is None:
        return None
    if not text or not text.strip():
        print("No text to synthesize; skipping TTS.", file=sys.stderr)
        return None

    tmp_wav = out_base.with_suffix(".wav")
    print(f"Synthesizing {out_base.stem}.{fmt}", file=sys.stderr)
    tts_api.tts_to_file(text=text, file_path=str(tmp_wav))

    final_path = out_base.with_suffix(f".{fmt}")
    if fmt == "wav":
        # Just rename the temporary wav to the final name
        try:
            tmp_wav.rename(final_path)
        except Exception:
            # If cross-filesystem or similar, fallback to copy
            import shutil
            shutil.copyfile(tmp_wav, final_path)
            tmp_wav.unlink(missing_ok=True)
        return final_path

    # Re-encode to requested format
    if fmt == "mp3":
        cmd = f'ffmpeg -y -i "{tmp_wav}" -vn -ar 44100 -b:a 160k "{final_path}"'
    elif fmt == "ogg":
        # Ogg Vorbis, mono, 22050 Hz
        cmd = f'ffmpeg -y -i "{tmp_wav}" -vn -ac 1 -ar 22050 -c:a libvorbis -qscale:a 5 "{final_path}"'
    elif fmt in ("opus"):
        # Opus in OGG container
        cmd = f'ffmpeg -y -i "{tmp_wav}" -vn -c:a libopus -b:a 96k "{final_path}"'
    elif fmt == "m4a":
        cmd = f'ffmpeg -y -i "{tmp_wav}" -vn -c:a aac -b:a 160k "{final_path}"'
    else:
        raise ValueError(f"Unsupported audio format: {fmt}")

    rc = os.system(cmd)
    if rc != 0:
        print(f"ffmpeg re-encode failed (rc={rc}); keeping WAV at {tmp_wav}", file=sys.stderr)
        return tmp_wav

    tmp_wav.unlink(missing_ok=True)
    return final_path


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
    parser.add_argument("--audio-format",
                        choices=["wav", "mp3", "ogg", "opus", "m4a"],
                        default="wav",
                        help="Output audio format (default: wav)")
    parser.add_argument("--out-prefix", default="/data/output/output",
                        help="Prefix for output files (default: /data/output/output)")
    args = parser.parse_args()

    input_path = Path(args.input)
    files = gather_inputs(input_path)

    # Prepare translators (expects fr->en and en->es installed in image)
    print("Preparing translators FR -> EN -> ES…", file=sys.stderr)
    fr_en, en_es = ensure_argos_translators_via_english(args.src, args.tgt)

    # Load Whisper once
    print(f"Loading Whisper model ({args.model_size}, device={args.device}, compute_type={args.compute_type})…", file=sys.stderr)
    model = WhisperModel(args.model_size, device=args.device, compute_type=args.compute_type)

    # Optional TTS (loaded once)
    tts_api = init_tts_or_none(not args.no_tts)

    for idx, f in enumerate(files, 1):
        print(f"\n--- [{idx}/{len(files)}] Processing {f.name} ---", file=sys.stderr)

        fr_text = transcribe_file(model, f, args.src, args.beam_size)
        if not fr_text or not fr_text.strip():
            print(f"No speech detected in {f.name}; skipping translation and TTS.", file=sys.stderr)
            continue

        # Write French transcript (will be cleaned up after)
        out_fr = Path(f"{args.out_prefix}/{f.stem}.fr.txt")
        out_fr.parent.mkdir(parents=True, exist_ok=True)
        out_fr.write_text(fr_text + "\n", encoding="utf-8")
        print(f"Wrote: {out_fr}", file=sys.stderr)

        # Translate via EN pivot
        es_text = translate_text(fr_text, fr_en, en_es)
        if not es_text or not es_text.strip():
            print(f"Empty translation for {f.name}; skipping TTS.", file=sys.stderr)
            # cleanup .fr
            try:
                if out_fr.exists():
                    out_fr.unlink()
                    print(f"Removed {out_fr}", file=sys.stderr)
            except Exception as e:
                print(f"Cleanup failed (.fr) for {f.name}: {e}", file=sys.stderr)
            continue

        # Write Spanish text (will be cleaned up after)
        out_es = Path(f"{args.out_prefix}/{f.stem}.es.txt")
        out_es.write_text(es_text + "\n", encoding="utf-8")
        print(f"Wrote: {out_es}", file=sys.stderr)

        # Optional Spanish speech
        if tts_api is not None:
            out_base = Path(f"{args.out_prefix}/{f.stem}.es")
            final_audio = maybe_synthesize_tts(tts_api, es_text, out_base, args.audio_format)
            if final_audio:
                print(f"Wrote: {final_audio}", file=sys.stderr)

        # Cleanup .txt artifacts
        try:
            if out_fr.exists():
                out_fr.unlink()
                print(f"Removed {out_fr}", file=sys.stderr)
            if out_es.exists():
                out_es.unlink()
                print(f"Removed {out_es}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: cleanup failed for {f.name}: {e}", file=sys.stderr)

    print("\nAll done.", file=sys.stderr)


if __name__ == "__main__":
    main()