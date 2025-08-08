import os
from pathlib import Path
from audio_processing import AudioProcessing

def debug_audio_processing():
    """Debug the audio processing step by step"""
    
    # Replace these paths with your actual paths
    input_directory = input("Enter your input directory path: ").strip()
    noise_profile = input("Enter your noise profile file path (or press Enter to skip): ").strip()
    
    if not noise_profile:
        noise_profile = None
    
    print(f"\n=== DEBUG INFO ===")
    print(f"Input directory: {input_directory}")
    print(f"Input dir exists: {os.path.exists(input_directory)}")
    print(f"Noise profile: {noise_profile}")
    if noise_profile:
        print(f"Noise profile exists: {os.path.exists(noise_profile)}")
    
    # Create processor
    try:
        processor = AudioProcessing(input_directory)
        print(f"Processor created successfully")
        print(f"Root path: {processor.root_path}")
        print(f"Combined path: {processor.combined_path}")
        print(f"Filtered path: {processor.filtered_path}")
    except Exception as e:
        print(f"ERROR creating processor: {e}")
        return
    
    # Check if combined files exist
    print(f"\n=== CHECKING FOR COMBINED FILES ===")
    combined_files = list(processor.combined_path.rglob("*.wav"))
    print(f"Found {len(combined_files)} combined files:")
    for f in combined_files:
        print(f"  - {f}")
    
    if not combined_files:
        print("No combined files found. Running full process...")
        success = processor.process_directory(
            input_dir=input_directory,
            output_dir=None,
            noise_profile_path=noise_profile
        )
        print(f"Process result: {success}")
    else:
        print("Combined files exist. Testing filtering only...")
        if noise_profile:
            print(f"Setting noise profile to: {noise_profile}")
            print(f"Noise profile exists: {os.path.exists(noise_profile)}")
            
            # Instead of using find_wave_files, let's create the structure from combined files
            print("\n=== CREATING STRUCTURE FROM COMBINED FILES ===")
            wav_files = {}
            for combined_file in combined_files:
                # Parse the path: combined/Cohort7/Session 20250805_161750.wav
                parts = combined_file.parts
                if len(parts) >= 2:
                    cohort = parts[-2]  # Cohort7
                    session_file = parts[-1]  # Session 20250805_161750.wav
                    session = session_file.replace('.wav', '')  # Session 20250805_161750
                    
                    if cohort not in wav_files:
                        wav_files[cohort] = {}
                    wav_files[cohort][session] = ['dummy.wav']  # We don't need the original files for filtering
                    
            print(f"Created wave files structure from combined files: {wav_files}")
            
            # Set the noise profile path
            processor.noise_profile_path = noise_profile
            print(f"Processor noise_profile_path set to: {processor.noise_profile_path}")
            
            # Test filtering with detailed noise profile checking
            print("\n=== CHECKING NOISE PROFILE DETAILS ===")
            noise_path = Path(noise_profile)
            print(f"Noise path: {noise_path}")
            print(f"Is file: {noise_path.is_file()}")
            print(f"Is directory: {noise_path.is_dir()}")
            
            if noise_path.is_dir():
                wav_files_in_noise_dir = list(noise_path.glob("*.wav"))
                print(f"WAV files in noise directory: {wav_files_in_noise_dir}")
            
            print("\n=== RUNNING FILTERING TEST ===")
            # Let's manually call the filtering with direct prints
            print("Manually calling filtering process...")
            
            # Check noise profile
            noise_path = Path(noise_profile)
            if noise_path.is_dir():
                wav_files_in_dir = list(noise_path.glob("*.wav"))
                if wav_files_in_dir:
                    noise_file_path = wav_files_in_dir[0]
                    print(f"Using noise file: {noise_file_path}")
                    
                    # Process each combined file
                    for combined_file in combined_files:
                        parts = combined_file.parts
                        if len(parts) >= 2:
                            cohort = parts[-2]  # Cohort7
                            session_file = parts[-1]  # Session 20250805_161750.wav
                            session = session_file.replace('.wav', '')  # Session 20250805_161750
                            
                            print(f"\nProcessing: {combined_file}")
                            output_file = processor.filtered_path / cohort / f"{session}_filtered.wav"
                            output_file.parent.mkdir(parents=True, exist_ok=True)
                            print(f"Output will be: {output_file}")
                            
                            # Try to process the audio file with detailed debugging
                            try:
                                print(f"  Loading audio: {combined_file}")
                                import librosa
                                import scipy.io.wavfile as wavfile
                                import noisereduce as nr
                                from scipy.signal import butter, filtfilt
                                
                                # Load audio and noise files
                                audio, sr = librosa.load(str(combined_file), sr=None)
                                noise, _ = librosa.load(str(noise_file_path), sr=sr)
                                print(f"  Loaded audio - shape: {audio.shape}, sample rate: {sr}, dtype: {audio.dtype}")
                                print(f"  Loaded noise - shape: {noise.shape}, dtype: {noise.dtype}")
                        
                                # Apply high-pass filter
                                nyquist = sr / 2
                                high_cutoff = 10_000 / nyquist
                                b, a = butter(4, high_cutoff, btype="highpass", analog=False, output="ba")
                                audio_filtered = filtfilt(b, a, audio)
                                noise_filtered = filtfilt(b, a, noise)
                                print(f"  Filtered audio - dtype: {audio_filtered.dtype}, shape: {audio_filtered.shape}")

                                # Apply noise reduction
                                print("  Applying noise reduction...")
                                cleaned_audio = nr.reduce_noise(
                                    y=audio_filtered,
                                    sr=sr,
                                    y_noise=noise_filtered,
                                    prop_decrease=0.7,
                                    chunk_size=512,
                                )
                                print(f"  Cleaned audio - dtype: {cleaned_audio.dtype}, shape: {cleaned_audio.shape}")
                                print(f"  Audio range: min={cleaned_audio.min()}, max={cleaned_audio.max()}")

                                # Convert to proper format for saving
                                if cleaned_audio.dtype != audio.dtype:
                                    print(f"  Converting from {cleaned_audio.dtype} to {audio.dtype}")
                                
                                # Try different approaches to save
                                print(f"  Attempting to save to: {output_file}")
                                try:
                                    # Method 1: Original approach
                                    wavfile.write(str(output_file), sr, cleaned_audio.astype(audio.dtype))
                                    print("  Saved using original method")
                                except Exception as save_error:
                                    print(f"  Original save method failed: {save_error}")
                                    try:
                                        # Method 2: Convert to int16
                                        import numpy as np
                                        if cleaned_audio.dtype == np.float32 or cleaned_audio.dtype == np.float64:
                                            # Convert float to int16
                                            audio_int16 = (cleaned_audio * 32767).astype(np.int16)
                                            wavfile.write(str(output_file), sr, audio_int16)
                                            print("  Saved using int16 conversion")
                                        else:
                                            raise save_error
                                    except Exception as save_error2:
                                        print(f"  Int16 save method also failed: {save_error2}")
                                        # Method 3: Use librosa
                                        import soundfile as sf
                                        sf.write(str(output_file), cleaned_audio, sr)
                                        print("  Saved using soundfile")
                                
                                # Check if file was actually created
                                if output_file.exists():
                                    file_size = output_file.stat().st_size
                                    print(f"✅ Successfully processed {session} - File created: {file_size} bytes")
                                else:
                                    print(f"❌ Processing completed but file still doesn't exist: {output_file}")
                                    
                            except Exception as e:
                                print(f"❌ Error processing {session}: {e}")
                                import traceback
                                print(traceback.format_exc())
                else:
                    print("No WAV files found in noise directory")
            else:
                print("Noise profile is not a directory")
            
            processor._process_audio_files(wav_files)
            print("=== FILTERING TEST COMPLETE ===")
        else:
            print("No noise profile provided - cannot test filtering")
    
    # Check results with more detail
    print(f"\n=== CHECKING RESULTS ===")
    print(f"Filtered path exists: {processor.filtered_path.exists()}")
    if processor.filtered_path.exists():
        # Check all subdirectories
        for item in processor.filtered_path.rglob("*"):
            if item.is_file():
                print(f"Found file: {item} ({item.stat().st_size} bytes)")
            elif item.is_dir():
                print(f"Found directory: {item}")
        
        filtered_files = list(processor.filtered_path.rglob("*.wav"))
        print(f"Found {len(filtered_files)} filtered files:")
        for f in filtered_files:
            size = f.stat().st_size
            print(f"  - {f} ({size} bytes)")
    else:
        print("Filtered directory doesn't exist")

if __name__ == "__main__":
    debug_audio_processing()
