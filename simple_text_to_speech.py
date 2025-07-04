#!/usr/bin/env python3
"""
Simple Text-to-Speech using CSM-1B
Basic single-text to speech conversion script
"""
import os
import time
import torch
import torchaudio
from generator import load_csm_1b

# Environment setup for Mac compatibility
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["FORCE_CPU"] = "1"  # Force CPU for stability

# Create output directory if it doesn't exist
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
  # 🎯 Configuration options
  text_to_convert = "CSM is a speech generation model from Sesame that generates RVQ audio codes from text and audio input! The model architecture employs a Llama backbone and a smaller audio decoder that produces Mimi audio codes."
  
  # 🎭 Voice settings
  speaker_id = 1  # 🔄 Choose speaker (0, 1, 2) for different voice characteristics
  
  print("🎤 Loading CSM-1B model...")
  model_start_time = time.time()
  generator = load_csm_1b(device="cpu")
  model_load_time = time.time() - model_start_time
  print(f"✅ Model loaded successfully! (⏱️ Loading time: {model_load_time:.2f}s)")
  
  print(f"🔄 Converting text: {text_to_convert}")
  print(f"🎭 Using speaker ID: {speaker_id}")
  
  # Generate speech with timing
  tts_start_time = time.time()
  audio_tensor = generator.generate(
    text=text_to_convert,
    speaker=speaker_id,  # Use specified speaker ID
    context=[],  # No context (for basic voice variation)
    max_audio_length_ms=10_000,  # 10 seconds max
  )
  tts_time = time.time() - tts_start_time
  
  # Save to file with speaker info in output directory
  output_file = os.path.join(OUTPUT_DIR, f"output_speech_speaker{speaker_id}.wav")
  torchaudio.save(
    output_file,
    audio_tensor.unsqueeze(0).cpu(),
    generator.sample_rate
  )
  
  # Performance summary
  audio_duration = len(audio_tensor) / generator.sample_rate
  total_time = model_load_time + tts_time
  
  print(f"✅ Speech saved to: {output_file}")
  print(f"📊 Audio duration: {audio_duration:.2f} seconds")
  print(f"📁 Output directory: {OUTPUT_DIR}/")
  print(f"\n⏱️ Performance Summary:")
  print(f"  • Model loading time: {model_load_time:.2f}s")
  print(f"  • Text-to-speech time: {tts_time:.2f}s")
  print(f"  • Total processing time: {total_time:.2f}s")
  print(f"  • Real-time factor: {tts_time/audio_duration:.2f}x")
  print(f"\n💡 Tips:")
  print(f"  • Change 'speaker_id' (0-2) for different voices")
  print(f"  • For advanced voice styles, use custom voice prompts")
  print(f"  • Try different speaker IDs to find your preferred voice")

if __name__ == "__main__":
  main() 