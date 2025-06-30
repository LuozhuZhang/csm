#!/usr/bin/env python3
"""
Simple Text-to-Speech using CSM-1B
Basic single-text to speech conversion script
"""
import os
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
  generator = load_csm_1b(device="cpu")
  print("✅ Model loaded successfully!")
  
  print(f"🔄 Converting text: {text_to_convert}")
  print(f"🎭 Using speaker ID: {speaker_id}")
  
  # Generate speech
  audio_tensor = generator.generate(
    text=text_to_convert,
    speaker=speaker_id,  # Use specified speaker ID
    context=[],  # No context (for basic voice variation)
    max_audio_length_ms=10_000,  # 10 seconds max
  )
  
  # Save to file with speaker info in output directory
  output_file = os.path.join(OUTPUT_DIR, f"output_speech_speaker{speaker_id}.wav")
  torchaudio.save(
    output_file,
    audio_tensor.unsqueeze(0).cpu(),
    generator.sample_rate
  )
  
  print(f"✅ Speech saved to: {output_file}")
  print(f"📊 Audio duration: {len(audio_tensor)/generator.sample_rate:.2f} seconds")
  print(f"📁 Output directory: {OUTPUT_DIR}/")
  print(f"\n💡 Tips:")
  print(f"  • Change 'speaker_id' (0-2) for different voices")
  print(f"  • For advanced voice styles, use custom voice prompts")
  print(f"  • Try different speaker IDs to find your preferred voice")

if __name__ == "__main__":
  main() 