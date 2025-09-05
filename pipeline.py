#!/usr/bin/env python3
"""
French audio -> Spanish text (and optional Spanish speech)
- Input: single file OR directory of files
- ASR: faster-whisper
- Translation: Argos Translate (FR -> EN -> ES)
- Optional TTS: Coqui TTS (Spanish)
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
    installed = argo.get_installed_languages()
    def get_lang(code): return next((l for l in installed if l.code == code), None)
    src, en, tgt = get_lang(src_code), get_lang("en"), get_lang(tgt_code)
    if not (src and en and tgt):
        raise SystemExit("Missing Argos packages (need fr->en and en->es installed).")
    return src.get_translation(en), en.get_translation(tgt)


def init_tts_or_none(enable_tts: bool):
    if not enable_tts:
        return None
    try:
        from TTS.api import TTS
        return TTS(model_name="tts_models/es/css10/vits")
    except Exception as e:
        print(f"TTS init failed: {e}", file=sys.stderr)
        return None


def transcribe_file(model, path: Path, src_lang: str, beam_size: int) -> str:
    print(f"Transcribing {path.name}", file=sys.stderr)
    segments, _ = model.transcribe(
        str(path), language=src_lang, task="transcribe",
        beam_size=beam_size, vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


def translate_text(fr_text: str, fr_en, en_es) -> str:
    en_text = fr_en.translate(fr_text)
    return en_es.translate(en_text)


def maybe_synthesize_tts(tts_api, text: str, out_base: Path, fmt: str):
    """Always synthesize to wav first, re-encode if needed"""
    tmp_wav = out_base.with_suffix(".wav")
    if tts_api is None:
        return
    print(f"Synthesizing {out_base.stem}.{fmt}", file=sys.stderr)
    tts_api.tts_to_file(text=text, file_path=str(tmp_wav))
    final_path = out_base.with_suffix(f".{fmt}")
    if fmt == "wav":
        tmp_wav.rename(final_path)
    else:
        if fmt == "mp3":
            cmd = f'ffmpeg -y -i "{tmp_wav}" -vn -ar 44100 -b:a 160k "{final_path}"'
        elif fmt in ("ogg", "opus"):
            cmd = f'ffmpeg -y -i "{tmp_wav}" -vn -c:a libopus -b:a 96k "{final_path}"'
        elif fmt == "m4a":
            cmd = f'ffmpeg -y -i "{tmp_wav}" -vn -c:a aac -b:a 160k "{final_path}"'
        else:
            raise ValueError(f"Unsupported audio format: {fmt}")
        os.system(cmd)
        tmp_wav.unlink(missing_ok=True)
    return final_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Audio file OR folder")
    parser.add_argument("--src", default="fr")
    parser.add_argument("--tgt", default="es")
    parser.add_argument("--model-size", dest="model_size", default="small")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--audio-format",
                        choices=["wav", "mp3", "ogg", "opus", "m4a"],
                        default="wav",
                        help="Output audio format (default: wav)")
    parser.add_argument("--out-prefix", default="/data/output/output")
    args = parser.parse_args()

    files = gather_inputs(Path(args.input))
    fr_en, en_es = ensure_argos_translators_via_english(args.src, args.tgt)
    model = WhisperModel(args.model_size, device=args.device, compute_type=args.compute_type)
    tts_api = init_tts_or_none(not args.no_tts)

    for f in files:
        print(f"--- Processing {f.name} ---", file=sys.stderr)
        fr_text = transcribe_file(model, f, args.src, args.beam_size)

        out_fr = Path(f"{args.out_prefix}/{f.stem}.fr.txt")
        out_fr.parent.mkdir(parents=True, exist_ok=True)
        out_fr.write_text(fr_text + "\n", encoding="utf-8")
        print(f"Wrote: {out_fr}", file=sys.stderr)

        es_text = translate_text(fr_text, fr_en, en_es)
        out_es = Path(f"{args.out_prefix}/{f.stem}.es.txt")
        out_es.write_text(es_text + "\n", encoding="utf-8")
        print(f"Wrote: {out_es}", file=sys.stderr)

        if tts_api is not None:
            out_base = Path(f"{args.out_prefix}/{f.stem}.es")
            final_audio = maybe_synthesize_tts(tts_api, es_text, out_base, args.audio_format)
            if final_audio:
                print(f"Wrote: {final_audio}", file=sys.stderr)

        # Cleanup text files if you don't want them
        try:
            if out_fr.exists(): out_fr.unlink()
            if out_es.exists(): out_es.unlink()
        except Exception as e:
            print(f"Cleanup failed: {e}", file=sys.stderr)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()