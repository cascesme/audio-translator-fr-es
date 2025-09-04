"""
Download and install offline models during build:

1) Argos Translate FR->ES package
2) Coqui TTS Spanish model

This makes first run snappy and usable offline.
"""
import os

# ---- Argos Translate: Install FR->ES package ----
import argostranslate.package
import argostranslate.translate

# Pull the latest package index and install FR->ES
argostranslate.package.update_package_index()
available = argostranslate.package.get_available_packages()
fr_es = [p for p in available if p.from_code == "fr" and p.to_code == "es"]
if fr_es:
    pkg = fr_es[0]
    download_path = pkg.download()
    argostranslate.package.install_from_path(download_path)

# Sanity: initialize translator once
_ = argostranslate.translate.get_installed_languages()

# ---- Coqui TTS: Pre-download a Spanish model ----
# We'll use the stable Spanish CSS10 VITS model.
# Model name: "tts_models/es/css10/vits"
try:
    from TTS.api import TTS
    # This will download artifacts to a cache dir during build
    TTS(model_name="tts_models/es/css10/vits")
except Exception as e:
    # Don’t fail the build if model fetch hiccups; user can still pass --no-tts
    print("Warning: Coqui TTS model prefetch failed:", e)
