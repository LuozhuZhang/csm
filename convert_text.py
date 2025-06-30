#!/usr/bin/env python3
"""
Easy Text-to-Speech Converter
Batch text-to-speech conversion tool for multiple texts
"""
import os
import torch
import torchaudio
from generator import load_csm_1b

# Mac compatibility
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["FORCE_CPU"] = "1"

# 📝 Edit your texts here - add as many as you want
TEXTS_TO_CONVERT = [
  "Hello, welcome to the future of artificial intelligence.",
  "This is CSM one B, a conversational speech model.",
  "You can easily convert any English text to natural speech.",
]

def convert_single_text(generator, text, output_filename):
  """Convert single text to speech with default voice settings"""
  print(f"🔄 Converting: {text[:50]}...")
  
  audio_tensor = generator.generate(
    text=text,
    speaker=0,  # Use default speaker (speaker 0)
    context=[],  # No voice context for batch processing
    max_audio_length_ms=15_000,  # 15 seconds max
  )
  
  torchaudio.save(
    output_filename,
    audio_tensor.unsqueeze(0).cpu(),
    generator.sample_rate
  )
  
  duration = len(audio_tensor) / generator.sample_rate
  print(f"✅ Saved: {output_filename} ({duration:.1f}s)")

def main():
  print("🎤 CSM-1B Batch Text-to-Speech Converter")
  print("=" * 40)
  
  # Load model once for efficiency
  print("📱 Loading model...")
  generator = load_csm_1b(device="cpu")
  print("✅ Model ready!")
  
  # Convert each text in the list
  for i, text in enumerate(TEXTS_TO_CONVERT, 1):
    output_file = f"speech_{i:02d}.wav"
    convert_single_text(generator, text, output_file)
  
  print(f"\n🎉 Converted {len(TEXTS_TO_CONVERT)} texts successfully!")
  print("📁 Output files: speech_01.wav, speech_02.wav, ...")

if __name__ == "__main__":
  main() 