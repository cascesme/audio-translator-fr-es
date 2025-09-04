import argparse
import os
import sys
from pathlib import Path

# ASR
from faster_whisper import WhisperModel

# Translation
import argostranslate.translate as argo

# Optional TTS
def synthesize_tts(text: str, out_wav: str):
    from TTS.api import TTS
    tts = TTS(model_name="tts_models/es/css10/vits")
    tts.tts_to_file(text=text, file_path=out_wav, speaker=None, language=None)

def main():
    parser = argparse.ArgumentParser(
        description="French audio -> Spanish text (and optional Spanish speech)"
    )
    parser.add_argument("audio", help="Path to input audio/video file (FR speech)")
    parser.add_argument("--src", default="fr", help="Source language (default: fr)")
    parser.add_argument("--tgt", default="es", help="Target language (default: es)")
    parser.add_argument("--model-size", dest="model_size", default="small",
                        help="Whisper size: tiny|base|small|medium|large-v3 (default: small)")
    parser.add_argument("--beam-size", type=int, default=5, help="ASR beam size (default: 5)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device (default: cpu)")
    parser.add_argument("--compute-type", default="int8", help="CTranslate2 compute type")
    parser.add_argument("--no-tts", action="store_true", help="Disable Spanish TTS")
    parser.add_argument("--out-prefix", default="output", help="Prefix for output files (default: output)")
    args = parser.parse_args()

    in_path = Path(args.audio)
    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    # ---------- 1) ASR ----------
    print("Loading Whisper model…", file=sys.stderr)
    model = WhisperModel(
        args.model_size,
        device=args.device,
        compute_type=args.compute_type
    )
    print("Transcribing…", file=sys.stderr)
    segments, info = model.transcribe(
        str(in_path),
        language=args.src,
        task="transcribe",
        beam_size=args.beam_size,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    fr_text = " ".join(seg.text.strip() for seg in segments)

    out_fr_txt = f"{args.out_prefix}.fr.txt"
    with open(out_fr_txt, "w", encoding="utf-8") as f:
        f.write(fr_text + "\n")
    print(f"Wrote French transcript: {out_fr_txt}")

    # ---------- 2) Translation ----------
    print("Translating FR -> ES…", file=sys.stderr)
    installed = argo.get_installed_languages()
    src_lang = next((l for l in installed if l.code == args.src), None)
    tgt_lang = next((l for l in installed if l.code == args.tgt), None)
    translator = src_lang.get_translation(tgt_lang)
    es_text = translator.translate(fr_text)

    out_es_txt = f"{args.out_prefix}.es.txt"
    with open(out_es_txt, "w", encoding="utf-8") as f:
        f.write(es_text + "\n")
    print(f"Wrote Spanish translation: {out_es_txt}")

    # ---------- 3) Optional TTS ----------
    if not args.no_tts:
        try:
            print("Synthesizing Spanish speech…", file=sys.stderr)
            out_es_wav = f"{args.out_prefix}.es.wav"
            synthesize_tts(es_text, out_es_wav)
            print(f"Wrote Spanish audio: {out_es_wav}")
        except Exception as e:
            print(f"Spanish TTS failed: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
