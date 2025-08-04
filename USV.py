import os
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, spectrogram
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt 
import logging
from typing import List, Optional, Tuple
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from scipy.io import wavfile
import glob

class waveProcessing:
    """Handles audio file consolidation and filtering for scientific experiments."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.logger = self._setup_logging()
        self.console = Console()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        return logging.getLogger(__name__)

    def find_wav_files(self, root_directory: str) -> List[str]:
        """
        Recursively find all WAV files in directory and subdirectories.

        Args:
            root_directory: Path to the root directory to search

        Returns:
            List of WAV file paths
        """
        wav_files = []
        for root, dirs, files in os.walk(root_directory):
            wav_pattern = os.path.join(root, "*.wav")
            wav_files.extend(glob.glob(wav_pattern))

        self.logger.info(f"Found {len(wav_files)} WAV files in {root_directory}")
        return sorted(wav_files)

    def install_requirements(self):
        """
            Print required packages. Run this if you get import errors.
        """  
        print("Required packages:")
        print("pip install numpy scipy matplotlib")
        print("or")
        print("conda install numpy scipy matplotlib")

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        """
            Design a Butterworth band-pass filter.
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        """
        Apply band-pass filter to data.
            """
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    def spectral_subtraction(self, signal, noise, alpha=2.0, beta=0.01):
        """
        Perform spectral subtraction to remove background noise.

        Args:
        signal: Input audio signal
        noise: Background noise signal
        alpha: Over-subtraction factor (default: 2.0)
        beta: Spectral floor factor (default: 0.01)

        Returns:
            Cleaned audio signal
        """
        # Ensure signals are the same length
        min_len = min(len(signal), len(noise))
        signal = signal[:min_len]
        noise = noise[:min_len]

        # Convert to frequency domain
        signal_fft = fft(signal)
        noise_fft = fft(noise)

        # Calculate magnitude spectra
        signal_mag = np.abs(signal_fft)
        noise_mag = np.abs(noise_fft)

        # Estimate noise magnitude (average over time)
        noise_mag_est = np.mean(noise_mag)

        # Spectral subtraction
        subtracted_mag = signal_mag - alpha * noise_mag_est

        # Apply spectral floor
        spectral_floor = beta * signal_mag
        subtracted_mag = np.maximum(subtracted_mag, spectral_floor)

        # Reconstruct signal with original phase
        signal_phase = np.angle(signal_fft)
        cleaned_fft = subtracted_mag * np.exp(1j * signal_phase)

        # Convert back to time domain
        cleaned_signal = np.real(ifft(cleaned_fft))

        return cleaned_signal

    def process_audio_file(self, audio_path, background_path, output_path, 
                        lowcut=20000, highcut=100000, sample_rate=None):
        """
        Process a single audio file: spectral subtraction + band-pass filtering.

        Args:
            audio_path: Path to input audio file
            background_path: Path to background noise file
            output_path: Path to save processed audio
            lowcut: Low frequency cutoff for band-pass filter (Hz)
            highcut: High frequency cutoff for band-pass filter (Hz)
            sample_rate: Target sample rate (None to keep original)
        """
        try:
            # Load audio files
            sr_signal, signal = wavfile.read(audio_path)
            sr_noise, noise = wavfile.read(background_path)
        
            # Convert to float
            if signal.dtype == np.int16:
                signal = signal.astype(np.float32) / 32768.0
            elif signal.dtype == np.int32:
                signal = signal.astype(np.float32) / 2147483648.0
        
            if noise.dtype == np.int16:
                noise = noise.astype(np.float32) / 32768.0
            elif noise.dtype == np.int32:
                noise = noise.astype(np.float32) / 2147483648.0
        
            # Handle stereo to mono conversion
            if len(signal.shape) > 1:
                signal = np.mean(signal, axis=1)
            if len(noise.shape) > 1:
                noise = np.mean(noise, axis=1)
        
            # Ensure same sample rate
            if sr_signal != sr_noise:
                print(f"Warning: Sample rate mismatch. Signal: {sr_signal}Hz, Noise: {sr_noise}Hz")
                print("Using signal sample rate for processing.")
        
            fs = sr_signal
        
            # Perform spectral subtraction
            print(f"  Applying spectral subtraction...")
            cleaned_signal = self.spectral_subtraction(signal, noise)
        
            # Apply band-pass filter for ultrasonic vocalizations
            print(f"  Applying band-pass filter ({lowcut}-{highcut} Hz)...")
        
            # Adjust filter frequencies if they exceed Nyquist frequency
            nyquist = fs / 2
            if highcut > nyquist:
                highcut = nyquist * 0.95  # 95% of Nyquist frequency
                print(f"  Adjusted high cutoff to {highcut:.0f} Hz (95% of Nyquist)")
        
            if lowcut > nyquist:
                lowcut = nyquist * 0.1  # 10% of Nyquist frequency
                print(f"  Adjusted low cutoff to {lowcut:.0f} Hz (10% of Nyquist)")
        
            filtered_signal = self.bandpass_filter(cleaned_signal, lowcut, highcut, fs)
        
            # Normalize to prevent clipping
            max_val = np.max(np.abs(filtered_signal))
            if max_val > 0:
                filtered_signal = filtered_signal / max_val * 0.95
        
            # Convert back to int16 for saving
            output_signal = (filtered_signal * 32767).astype(np.int16)
        
            # Save processed audio
            wavfile.write(output_path, fs, output_signal)

            return True, f"Successfully processed {audio_path.name}"
        except Exception as e:
            return False, f"Error processing {audio_path.name}: {str(e)}"
    
    def consolidate_audio_files(
        self, wav_files: List[str], output_path: str, gap_seconds: float = 0.5
    ) -> bool:
        """
        Consolidate multiple WAV files into a single file.

        Args:
            wav_files: List of WAV file paths
            output_path: Path for the consolidated output file
            gap_seconds: Silence gap between files in seconds

        Returns:
            True if successful, False otherwise
        """
        if not wav_files:
            self.logger.error("No WAV files provided for consolidation")
            return False

        consolidated_audio = []
        target_sample_rate = None

        # Create silence gap
        gap_samples = int(gap_seconds * self.sample_rate)
        silence_gap = np.zeros(gap_samples, dtype=np.float32)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                "Consolidating audio files...", total=len(wav_files)
            )

            for i, filepath in enumerate(wav_files):
                progress.update(
                    task, description=f"Processing {os.path.basename(filepath)}"
                )

                audio_data, sample_rate = self.load_audio_file(filepath)

                if audio_data is None:
                    continue

                # Set target sample rate from first file
                if target_sample_rate is None:
                    target_sample_rate = sample_rate

                # Resample if necessary
                if sample_rate != target_sample_rate:
                    audio_data = self._resample_audio(
                        audio_data, sample_rate, target_sample_rate
                    )

                # Convert stereo to mono if needed
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)

                consolidated_audio.append(audio_data)

                # Add gap between files (except for the last file)
                if i < len(wav_files) - 1:
                    consolidated_audio.append(silence_gap)

                progress.advance(task)

        if not consolidated_audio:
            self.logger.error("No audio data to consolidate")
            return False

        # Combine all audio data
        final_audio = np.concatenate(consolidated_audio)

        # Convert back to int16 for saving
        final_audio_int16 = (final_audio * 32767).astype(np.int16)

        # Save consolidated file
        try:
            wavfile.write(output_path, target_sample_rate, final_audio_int16)
            self.logger.info(f"Consolidated audio saved to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving consolidated audio: {e}")
            return False

    def process_directory(
            self,
            input_dir: str,
            output_dir: str,
            gap_seconds: float = 0.5,
            noise_profile_path: Optional[str] = None,
        ) -> bool:
            """
            Complete processing pipeline for a directory.

            Args:
                input_dir: Directory containing WAV files
                output_dir: Directory for output files
                preset: Filter preset to apply
                gap_seconds: Gap between consolidated files
                noise_profile_path: Path to noise profile file for noise reduction

            Returns:
                True if successful, False otherwise
            """
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            self.find_and_process_wav_files(input_dir, noise_profile_path, output_dir)

            self.console.print("[green]Processing completed successfully![/green]")
            return True

    def find_and_process_wav_files(self, root_directory, background_file, output_directory=None, 
                                lowcut=20000, highcut=100000):
        """
        Find all .wav files, consolidate them, and apply audio processing.
        
        Args:
        root_directory (str): Path to the root directory to search
        background_file (str): Path to background noise file
        output_directory (str, optional): Path to output directory
        lowcut (int): Low frequency cutoff for band-pass filter (Hz)
        highcut (int): High frequency cutoff for band-pass filter (Hz)
        """
        root_path = Path(root_directory)
        background_path = Path(background_file)
        
        if not root_path.exists():
            print(f"Error: Directory '{root_directory}' does not exist.")
            return

        if not background_path.exists():
            print(f"Error: Background file '{background_file}' does not exist.")
            return
        
        # Set up output directory
        if output_directory is None:
            output_path = root_path / "processed_wav"
        else:
            output_path = Path(output_directory)
        
        output_path.mkdir(exist_ok=True)
        
        # Dictionary to store wav files by folder
        wav_files_by_folder = {}
        
        # Walk through all subdirectories
        for folder_path in root_path.rglob('*'):
            if folder_path.is_dir():
                # Find all .wav files in this folder
                wav_files = list(folder_path.glob('*.wav'))
                
                if wav_files:
                    folder_name = folder_path.name
                    # Handle duplicate folder names by adding parent directory
                    relative_path = folder_path.relative_to(root_path)
                    unique_folder_name = str(relative_path).replace(os.sep, '_')
                    
                    wav_files_by_folder[unique_folder_name] = {
                        'source_path': folder_path,
                        'wav_files': wav_files
                    }
        
        # Process files
        total_files_processed = 0
        successful_files = 0
        
        print(f"Using background file: {background_path}")
        print(f"Band-pass filter range: {lowcut}-{highcut} Hz")
        print(f"Output directory: {output_path}")
        print("="*50)
        
        for folder_name, folder_info in wav_files_by_folder.items():
            # Create folder-specific output directory
            folder_output_path = output_path / folder_name
            folder_output_path.mkdir(exist_ok=True)
            
            print(f"\nProcessing folder: {folder_info['source_path']}")
            print(f"Found {len(folder_info['wav_files'])} .wav files")
            
            # Process each wav file
            for wav_file in folder_info['wav_files']:
                print(f"\nProcessing: {wav_file.name}")
                
                # Create output filename
                output_filename = f"processed_{wav_file.name}"
                output_file_path = folder_output_path / output_filename
                
                # Process the audio file
                success, message = self.process_audio_file(
                    wav_file, background_path, output_file_path, 
                    lowcut, highcut
                )
                
                print(f"  {message}")
                
                total_files_processed += 1
                if success:
                    successful_files += 1
        
        print(f"\n=== Summary ===")
        print(f"Total folders processed: {len(wav_files_by_folder)}")
        print(f"Total .wav files processed: {total_files_processed}")
        print(f"Successfully processed: {successful_files}")
        print(f"Failed: {total_files_processed - successful_files}")
        print(f"Output directory: {output_path}")

    def create_spectrogram(self, audio_file, output_file=None):
        """
        Create and save a spectrogram of the audio file.
        """
        try:
            sr, audio = wavfile.read(audio_file)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Create spectrogram
            f, t, Sxx = spectrogram(audio, sr, nperseg=1024)
            
            # Plot spectrogram
            plt.figure(figsize=(12, 8))
            plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.title(f'Spectrogram - {Path(audio_file).name}')
            plt.colorbar(label='Power [dB]')
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"Spectrogram saved to: {output_file}")
            else:
                plt.show()
                
        except Exception as e:
            print(f"Error creating spectrogram: {e}")

    def list_wav_files_only(self, root_directory):
        """
        Just list all .wav files found without processing them.
        """
        root_path = Path(root_directory)
        
        if not root_path.exists():
            print(f"Error: Directory '{root_directory}' does not exist.")
            return
        
        print(f"Searching for .wav files in: {root_directory}\n")
        
        total_files = 0
        for folder_path in root_path.rglob('*'):
            if folder_path.is_dir():
                wav_files = list(folder_path.glob('*.wav'))
                
                if wav_files:
                    print(f"Folder: {folder_path}")
                    for wav_file in wav_files:
                        print(f"  - {wav_file.name}")
                    total_files += len(wav_files)
                    print()
        
        print(f"Total .wav files found: {total_files}")
    
    def display_found_files(self, wav_files: List[str]):
        """Display a table of found WAV files."""
        table = Table(title="Found WAV Files")
        table.add_column("Index", style="cyan", no_wrap=True)
        table.add_column("File Name", style="magenta")
        table.add_column("Directory", style="green")

        for i, filepath in enumerate(wav_files, 1):
            filename = os.path.basename(filepath)
            directory = os.path.dirname(filepath)
            table.add_row(str(i), filename, directory)

        self.console.print(table)

if __name__ == "__main__":
    print("WAV File Processor with Spectral Subtraction and Ultrasonic Filtering")
    print("="*70)

    wav_processing = waveProcessing()
    
    # Check if required packages are available
    try:
        import numpy as np
        import scipy
        print("✓ Required packages are available")
    except ImportError:
        print("✗ Missing required packages")
        wav_processing.install_requirements()
        exit()
    
    # Get input parameters
    root_dir = input("\nEnter the root directory path to search: ").strip()
    
    if not root_dir:
        print("No directory specified. Exiting.")
        exit()
    
    # Ask user what they want to do
    print("\nChoose an option:")
    print("1. List .wav files only (preview)")
    print("2. Process .wav files (spectral subtraction + filtering)")
    print("3. Create spectrogram of a single file")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        wav_processing.list_wav_files_only(root_dir)
        
    elif choice == "2":
        # Get background file
        background_file = input("Enter path to background noise file: ").strip()
        if not background_file:
            print("Background file is required for processing. Exiting.")
            exit()
        
        # Get output directory
        output_dir = input("Enter output directory (or press Enter for default): ").strip()
        if not output_dir:
            output_dir = None
        
        # Get filter parameters
        print("\nUltrasonic vocalization band-pass filter settings:")
        lowcut = input("Enter low frequency cutoff (Hz) [default: 20000]: ").strip()
        highcut = input("Enter high frequency cutoff (Hz) [default: 100000]: ").strip()
        
        lowcut = int(lowcut) if lowcut else 20000
        highcut = int(highcut) if highcut else 100000
        
        wav_processing.find_and_process_wav_files(root_dir, background_file, output_dir, lowcut, highcut)
        
    elif choice == "3":
        audio_file = input("Enter path to audio file for spectrogram: ").strip()
        if audio_file and Path(audio_file).exists():
            output_spec = input("Enter output path for spectrogram (or press Enter to display): ").strip()
            wav_processing.create_spectrogram(audio_file, output_spec if output_spec else None)
        else:
            print("Invalid audio file path.")
            
    else:
        print("Invalid choice. Exiting.")


        