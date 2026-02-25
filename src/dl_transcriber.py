import os
import subprocess
import torch

def transcribe_audio_dl(audio_path, output_midi_path):
    """
    High-level function to run the Transkun Deep Learning transcription model.

    """
    
    # --- 1. Verify audio file path ---
    if not os.path.exists(audio_path):
        print(f"Error: File not found {audio_path}")
        return False

    # --- 2. Hardware Check ---
    print("\nStarting Transkun AI Transcriber...")
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected.")
        print("The process runs on CPU. Significant delays are expected.")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Detected GPU: {gpu_name}. Transcription will be faster.")

    # --- 3. Execute Transkun through Subprocess ---
    command = ["transkun", audio_path, output_midi_path]
    
    try:
        print(f"Processing: {os.path.basename(audio_path)}")
        
        # Execute command. Το check=True ensures that if Transkun "blows",
        # there will be an exception (CalledProcessError) that we will catch.
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        
        # --- 4. Verify ---
        if os.path.exists(output_midi_path):
            print(f"Success! MIDI was saved in: {output_midi_path}")
            return True
        else:
            print("Error: MIDI wasn't created")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"Critical error while running Transkun.")
        print(f"Error details: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: Transkun is not installed")
        print("Please run: pip install transkun")
        return False