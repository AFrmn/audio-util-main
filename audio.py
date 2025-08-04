import glob
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from scipy import signal
from scipy.io import wavfile

class AudioProcessor:
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

    def load_audio_file(self, filepath: str) -> Tuple[np.ndarray, int]:
        """
        Load a WAV file and return audio data and sample rate.

        Args:
            filepath: Path to the WAV file

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            sample_rate, audio_data = wavfile.read(filepath)

            # Convert to float32 and normalize
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype == np.uint8:
                audio_data = (audio_data.astype(np.float32) - 128) / 128.0

            return audio_data, sample_rate

        except Exception as e:
            self.logger.error(f"Error loading {filepath}: {e}")
            return None, None

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

    def _resample_audio(
        self, audio_data: np.ndarray, original_rate: int, target_rate: int
    ) -> np.ndarray:
        """Resample audio to target sample rate."""
        if original_rate == target_rate:
            return audio_data

        # Calculate resampling ratio
        ratio = target_rate / original_rate
        new_length = int(len(audio_data) * ratio)

        # Use scipy's resample function
        resampled = signal.resample(audio_data, new_length)
        return resampled.astype(np.float32)

    def apply_filter_preset(
        self,
        input_path: str,
        output_path: str,
        preset: str = "scientific",
        noise_profile_path: Optional[str] = None,
    ) -> bool:
        """
        Apply predefined filter presets to audio file.

        Args:
            input_path: Path to input audio file
            output_path: Path for filtered output file
            preset: Filter preset name
            noise_profile_path: Path to background noise file for noise reduction

        Returns:
            True if successful, False otherwise
        """
        audio_data, sample_rate = self.load_audio_file(input_path)

        if audio_data is None:
            return False

        # Apply filters based on preset
        if preset == "scientific":
            filtered_audio = self._apply_scientific_preset(
                audio_data, sample_rate, noise_profile_path
            )
        elif preset == "noise_reduction":
            filtered_audio = self._apply_noise_reduction_preset(
                audio_data, sample_rate, noise_profile_path
            )
        elif preset == "vocal_enhancement":
            filtered_audio = self._apply_vocal_enhancement_preset(
                audio_data, sample_rate, noise_profile_path
            )
        elif preset == "adaptive_noise_reduction":
            filtered_audio = self._apply_adaptive_noise_reduction_preset(
                audio_data, sample_rate, noise_profile_path
            )
        else:
            self.logger.warning(f"Unknown preset '{preset}', using scientific preset")
            filtered_audio = self._apply_scientific_preset(
                audio_data, sample_rate, noise_profile_path
            )

        # Convert back to int16 for saving
        filtered_audio_int16 = (filtered_audio * 32767).astype(np.int16)

        try:
            wavfile.write(output_path, sample_rate, filtered_audio_int16)
            self.logger.info(f"Filtered audio saved to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving filtered audio: {e}")
            return False

    def _apply_scientific_preset(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        noise_profile_path: Optional[str] = None,
    ) -> np.ndarray:
        """Apply scientific analysis optimized filters."""
        # Apply noise reduction first if noise profile is provided
        if noise_profile_path:
            audio_data = self._apply_spectral_subtraction(
                audio_data, sample_rate, noise_profile_path
            )

        # High-pass filter to remove low-frequency noise
        nyquist = sample_rate / 2
        high_cutoff = 50 / nyquist
        b_high, a_high = signal.butter(4, high_cutoff, btype="high")
        filtered = signal.filtfilt(b_high, a_high, audio_data)

        # Bandpass filter for typical experimental frequencies
        low_cutoff = 5000 / nyquist
        high_cutoff = 202000 / nyquist
        b_band, a_band = signal.butter(4, [low_cutoff, high_cutoff], btype="band")
        filtered = signal.filtfilt(b_band, a_band, filtered)

        # Normalize
        filtered = self._normalize_audio(filtered)

        return filtered

    def _apply_noise_reduction_preset(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        noise_profile_path: Optional[str] = None,
    ) -> np.ndarray:
        """Apply noise reduction filters."""
        # Apply spectral subtraction first if noise profile is provided
        if noise_profile_path:
            filtered = self._apply_spectral_subtraction(
                audio_data, sample_rate, noise_profile_path
            )
        else:
            filtered = audio_data.copy()

        # Notch filter for 50Hz/60Hz power line noise
        nyquist = sample_rate / 2

        # 50Hz notch filter
        b_notch50, a_notch50 = signal.iirnotch(50, 30, sample_rate)
        filtered = signal.filtfilt(b_notch50, a_notch50, filtered)

        # 60Hz notch filter
        b_notch60, a_notch60 = signal.iirnotch(60, 30, sample_rate)
        filtered = signal.filtfilt(b_notch60, a_notch60, filtered)

        # Low-pass filter to remove high-frequency noise
        low_cutoff = 10000 / nyquist
        b_low, a_low = signal.butter(6, low_cutoff, btype="low")
        filtered = signal.filtfilt(b_low, a_low, filtered)

        # Normalize
        filtered = self._normalize_audio(filtered)

        return filtered

    def _apply_vocal_enhancement_preset(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        noise_profile_path: Optional[str] = None,
    ) -> np.ndarray:
        """Apply vocal enhancement filters."""
        # Apply noise reduction first if noise profile is provided
        if noise_profile_path:
            filtered = self._apply_spectral_subtraction(
                audio_data, sample_rate, noise_profile_path
            )
        else:
            filtered = audio_data.copy()

        # Bandpass filter for human vocal range
        nyquist = sample_rate / 2
        low_cutoff = 300 / nyquist
        high_cutoff = 3400 / nyquist
        b_vocal, a_vocal = signal.butter(4, [low_cutoff, high_cutoff], btype="band")
        filtered = signal.filtfilt(b_vocal, a_vocal, filtered)

        # Normalize
        filtered = self._normalize_audio(filtered)

        return filtered

    def _apply_adaptive_noise_reduction_preset(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        noise_profile_path: Optional[str] = None,
    ) -> np.ndarray:
        """Apply adaptive noise reduction based on noise profile."""
        if not noise_profile_path:
            self.console.print(
                "[yellow]Warning: Adaptive noise reduction requires a noise profile. Using standard noise reduction.[/yellow]"
            )
            return self._apply_noise_reduction_preset(audio_data, sample_rate)

        # Apply advanced spectral subtraction
        filtered = self._apply_spectral_subtraction(
            audio_data, sample_rate, noise_profile_path, alpha=2.0, beta=0.01
        )

        # Apply Wiener filtering
        filtered = self._apply_wiener_filter(filtered, sample_rate, noise_profile_path)

        # Normalize
        filtered = self._normalize_audio(filtered)

        return filtered

    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio to prevent clipping."""
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val * 0.95
        return audio_data

    def _apply_spectral_subtraction(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        noise_profile_path: str,
        alpha: float = 1.5,
        beta: float = 0.1,
    ) -> np.ndarray:
        """
        Apply spectral subtraction using noise profile.

        Args:
            audio_data: Input audio signal
            sample_rate: Sample rate of the audio
            noise_profile_path: Path to noise profile file
            alpha: Over-subtraction factor (higher = more aggressive)
            beta: Spectral floor factor (prevents over-subtraction)

        Returns:
            Filtered audio data
        """
        # Load noise profile
        noise_data, noise_sample_rate = self.load_audio_file(noise_profile_path)
        if noise_data is None:
            self.console.print(
                f"[red]Failed to load noise profile: {noise_profile_path}[/red]"
            )
            return audio_data

        # Resample noise if necessary
        if noise_sample_rate != sample_rate:
            noise_data = self._resample_audio(
                noise_data, noise_sample_rate, sample_rate
            )

        # Convert to mono if stereo
        if len(noise_data.shape) > 1:
            noise_data = np.mean(noise_data, axis=1)

        # Parameters for STFT
        nperseg = 2048
        noverlap = nperseg // 2

        # Compute STFT of both signals
        f_signal, t_signal, stft_signal = signal.stft(
            audio_data, sample_rate, nperseg=nperseg, noverlap=noverlap
        )
        f_noise, t_noise, stft_noise = signal.stft(
            noise_data, sample_rate, nperseg=nperseg, noverlap=noverlap
        )

        # Compute noise power spectrum (average across time)
        noise_power = np.mean(np.abs(stft_noise) ** 2, axis=1, keepdims=True)

        # Compute signal power spectrum
        signal_power = np.abs(stft_signal) ** 2

        # Spectral subtraction
        # Estimate clean signal power
        clean_power = signal_power - alpha * noise_power

        # Apply spectral floor to prevent over-subtraction
        spectral_floor = beta * signal_power
        clean_power = np.maximum(clean_power, spectral_floor)

        # Compute gain function
        gain = np.sqrt(clean_power / (signal_power + 1e-10))

        # Apply gain to original signal
        enhanced_stft = stft_signal * gain

        # Convert back to time domain
        _, enhanced_signal = signal.istft(
            enhanced_stft, sample_rate, nperseg=nperseg, noverlap=noverlap
        )

        # Ensure output length matches input
        if len(enhanced_signal) > len(audio_data):
            enhanced_signal = enhanced_signal[: len(audio_data)]
        elif len(enhanced_signal) < len(audio_data):
            # Pad with zeros if needed
            enhanced_signal = np.pad(
                enhanced_signal, (0, len(audio_data) - len(enhanced_signal))
            )

        return enhanced_signal.astype(np.float32)

    def _apply_wiener_filter(
        self, audio_data: np.ndarray, sample_rate: int, noise_profile_path: str
    ) -> np.ndarray:
        """
        Apply Wiener filter for noise reduction.

        Args:
            audio_data: Input audio signal
            sample_rate: Sample rate of the audio
            noise_profile_path: Path to noise profile file

        Returns:
            Filtered audio data
        """
        # Load noise profile
        noise_data, noise_sample_rate = self.load_audio_file(noise_profile_path)
        if noise_data is None:
            return audio_data

        # Resample noise if necessary
        if noise_sample_rate != sample_rate:
            noise_data = self._resample_audio(
                noise_data, noise_sample_rate, sample_rate
            )

        # Convert to mono if stereo
        if len(noise_data.shape) > 1:
            noise_data = np.mean(noise_data, axis=1)

        # Parameters for STFT
        nperseg = 2048
        noverlap = nperseg // 2

        # Compute STFT
        f_signal, t_signal, stft_signal = signal.stft(
            audio_data, sample_rate, nperseg=nperseg, noverlap=noverlap
        )
        f_noise, t_noise, stft_noise = signal.stft(
            noise_data, sample_rate, nperseg=nperseg, noverlap=noverlap
        )

        # Estimate noise power spectrum
        noise_power = np.mean(np.abs(stft_noise) ** 2, axis=1, keepdims=True)

        # Estimate signal power spectrum
        signal_power = np.abs(stft_signal) ** 2

        # Wiener filter gain
        wiener_gain = signal_power / (signal_power + noise_power + 1e-10)

        # Apply Wiener filter
        filtered_stft = stft_signal * wiener_gain

        # Convert back to time domain
        _, filtered_signal = signal.istft(
            filtered_stft, sample_rate, nperseg=nperseg, noverlap=noverlap
        )

        # Ensure output length matches input
        if len(filtered_signal) > len(audio_data):
            filtered_signal = filtered_signal[: len(audio_data)]
        elif len(filtered_signal) < len(audio_data):
            filtered_signal = np.pad(
                filtered_signal, (0, len(audio_data) - len(filtered_signal))
            )

        return filtered_signal.astype(np.float32)

    def create_noise_profile(self, noise_files: List[str], output_path: str) -> bool:
        """
        Create a noise profile from multiple noise files.

        Args:
            noise_files: List of noise file paths
            output_path: Path to save the noise profile

        Returns:
            True if successful, False otherwise
        """
        if not noise_files:
            self.console.print("[red]No noise files provided[/red]")
            return False

        noise_segments = []
        target_sample_rate = None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                "Creating noise profile...", total=len(noise_files)
            )

            for filepath in noise_files:
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

                noise_segments.append(audio_data)
                progress.advance(task)

        if not noise_segments:
            self.console.print("[red]No valid noise data found[/red]")
            return False

        # Concatenate all noise segments
        noise_profile = np.concatenate(noise_segments)

        # Convert to int16 for saving
        noise_profile_int16 = (noise_profile * 32767).astype(np.int16)

        try:
            wavfile.write(output_path, target_sample_rate, noise_profile_int16)
            self.console.print(f"[green]Noise profile saved to {output_path}[/green]")
            return True
        except Exception as e:
            self.console.print(f"[red]Error saving noise profile: {e}[/red]")
            return False

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        preset: str = "scientific",
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

        # Find all WAV files
        wav_files = self.find_wav_files(input_dir)

        if not wav_files:
            self.console.print("[red]No WAV files found in directory[/red]")
            return False

        # Display found files
        self.display_found_files(wav_files)

        # Consolidate files
        consolidated_path = os.path.join(output_dir, "consolidated.wav")
        if not self.consolidate_audio_files(wav_files, consolidated_path, gap_seconds):
            return False

        # Apply filters
        filtered_path = os.path.join(output_dir, f"filtered_{preset}.wav")
        if not self.apply_filter_preset(
            consolidated_path, filtered_path, preset, noise_profile_path
        ):
            return False

        self.console.print("[green]Processing completed successfully![/green]")
        return True

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
