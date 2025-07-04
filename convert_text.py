#!/usr/bin/env python3
"""
Batch Text-to-Speech Converter using CSM-1B
Efficient batch processing with model reuse
"""
import os
import time
import torch
import torchaudio
from generator import load_csm_1b

# Mac compatibility
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["FORCE_CPU"] = "1"

# Create output directory if it doesn't exist
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
  # 📝 Texts to convert
  texts_to_convert = [
    "Hello, this is the first demonstration of CSM text to speech.",
    "CSM is a powerful speech generation model that creates high-quality audio.",
    "This batch conversion shows how to process multiple texts efficiently."
  ]
  
  # 🎭 Voice settings
  speaker_id = 2  # Choose speaker (0, 1, 2) for different voice characteristics
  
  print("🎤 Loading CSM-1B model...")
  model_start_time = time.time()
  generator = load_csm_1b(device="cpu")
  model_load_time = time.time() - model_start_time
  print(f"✅ Model loaded successfully! (⏱️ Loading time: {model_load_time:.2f}s)")
  
  print(f"🔄 Converting {len(texts_to_convert)} texts...")
  print(f"🎭 Using speaker ID: {speaker_id}")
  
  total_tts_time = 0
  total_audio_duration = 0
  
  # Convert each text
  tts_start_time = time.time()
  for i, text in enumerate(texts_to_convert, 1):
    print(f"\n📝 Processing text {i}/{len(texts_to_convert)}: {text[:50]}...")
    
    # Generate speech for current text
    single_start = time.time()
    audio_tensor = generator.generate(
      text=text,
      speaker=speaker_id,
      context=[],  # No context for basic voice variation
      max_audio_length_ms=10_000,  # 10 seconds max per text
    )
    single_time = time.time() - single_start
    
    # Save to file
    output_file = os.path.join(OUTPUT_DIR, f"speech_{i:02d}.wav")
    torchaudio.save(
      output_file,
      audio_tensor.unsqueeze(0).cpu(),
      generator.sample_rate
    )
    
    # Calculate metrics
    audio_duration = len(audio_tensor) / generator.sample_rate
    total_audio_duration += audio_duration
    
    print(f"✅ Saved: {output_file} ({audio_duration:.2f}s audio, {single_time:.2f}s processing)")
  
  total_tts_time = time.time() - tts_start_time
  total_time = model_load_time + total_tts_time
  
  # Performance summary
  print(f"\n⏱️ Performance Summary:")
  print(f"  • Model loading time: {model_load_time:.2f}s")
  print(f"  • Total TTS processing: {total_tts_time:.2f}s")
  print(f"  • Total processing time: {total_time:.2f}s")
  print(f"  • Total audio generated: {total_audio_duration:.2f}s")
  print(f"  • Average real-time factor: {total_tts_time/total_audio_duration:.2f}x")
  print(f"  • Files per second: {len(texts_to_convert)/total_tts_time:.2f}")
  
  print(f"\n✅ Batch conversion completed!")
  print(f"📁 All files saved to: {OUTPUT_DIR}/")
  print(f"🎵 Generated {len(texts_to_convert)} audio files")

if __name__ == "__main__":
  main() 