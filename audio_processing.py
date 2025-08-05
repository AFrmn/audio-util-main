import logging
import os
import wave
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.io.wavfile as wavfile  # cSpell:ignore wavfile
from rich.console import Console

# from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from scipy import signal
from scipy.signal import butter, sosfilt  # cSpell:ignore sosfilt


class AudioProcessing:
    def __init__(self, root_directory: Optional[str] = None):
        if root_directory is None:
            raise ValueError("root_directory cannot be None")
        self.root_path = Path(root_directory)
        self.combined_path = self.root_path / Path("combined")
        self.filtered_path = self.root_path / Path("filtered")
        self.npz_file = self.root_path / Path("noise/noise.npz")
        self.console = Console()
        self.logger = self._setup_logging()

        # Create paths for results directories
        Path(self.combined_path).mkdir(parents=True, exist_ok=True)

    def _combine_wave_file_list(
        self, file_list: list, cohort: str, session: str
    ) -> None:
        """
        Combines general wave file data for a cohort/session combo from all initial

        Args:
            file_list (list): List of file names under a session
            cohort (str): Name given to the cohort directory
            session (str): name given to the session directory
        """
        data = []
        for wave_file in file_list:
            with wave.open(
                str(os.path.join(self.root_path, cohort, session, wave_file)), "rb"
            ) as w:
                data.append([w.getparams(), w.readframes(w.getnframes())])

            out_path = self.combined_path / Path(cohort)
            out_path.mkdir(parents=True, exist_ok=True)

            out_file = out_path / Path(session + ".wav")
            with wave.open(str(out_file), "wb") as output:
                if data:  # Check if we have data
                    output.setparams(data[0][0])
                    for _, frames in data:
                        output.writeframes(frames)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        return logging.getLogger(__name__)

    def find_wave_files(self) -> dict:
        """
        Recursively find all WAV files in directory and subdirectories.

        Args:
            root_directory: Path to the root directory to search

        Returns:
            Dict of WAV file paths organized by cohort and session
        """
        wav_files = {}
        # There is without question a better way to do this loop, but this was how I was able to debug it
        for root, cohorts, _ in os.walk(str(self.root_path)):
            for cohort in cohorts:
                if cohort.startswith("Cohort"):
                    wav_files[cohort] = {}
                    for _, sessions, _ in os.walk(os.path.join(root, cohort)):
                        for session in sessions:
                            if session.startswith("Session"):
                                wav_files[cohort][session] = []
                                for _, _, files in os.walk(
                                    os.path.join(root, cohort, session)
                                ):
                                    for file in files:
                                        if file.endswith(".wav"):
                                            wav_files[cohort][session].append(file)

        self.logger.info(
            f"Found {len(wav_files.keys())} cohorts in {str(self.root_path)}"
        )

        # This needs to be sorted for easy debugging and I wanted to log out the results of each
        sorted_results = dict(sorted(wav_files.items()))
        for cohort, sessions in wav_files.items():
            self.logger.info(f"Found {len(sessions.keys())} sessions in {cohort}")
            sorted_results[cohort] = dict(sorted(sessions.items()))
            for session, files in sessions.items():
                self.logger.info(f"Found {len(files)} files in {session}")
        return sorted_results

    def _subtract_spectral_noise_background(self, wave_files: dict) -> None:
        # Load Noise Profile ONCE and Save as NZ-something
        noise_data = (
            self.load_noise_profile()
        )  # todo! this doesn't work; I need to extract the noise data not from an .npz
        noise_freqs, noise_psd = self.compute_noise_spectrum(noise_data)
        noise_floor_db = self.get_noise_floor_db(noise_psd)

        # Save as npz file (somehow)
        self.npz_file.mkdir(parents=True, exist_ok=True)
        self.save_noise_spectrum(noise_freqs, noise_psd, noise_floor_db)

        # Create the filtered path
        self.filtered_path.mkdir(parents=True, exist_ok=True)

        # Loop through consolidated files
        for cohort in wave_files.keys():
            for session in wave_files[cohort].keys():
                combined_file_path = os.path.join(
                    self.combined_path, cohort, session + ".wav"
                )

                # Open the file & turn the combined_file_path into whatever noise_data is so I can subtract it
                freqs, psd = self.compute_noise_spectrum(
                    noise_data
                )  # todo! swap this for whatever the right method is
                # Subtract the noise and the floor DB
                # Apply the Bandpass UltraSonic (US) filter

                # Output each new filtered file
                filtered_file_path = os.path.join(
                    self.filtered_path, cohort, session + ".wav"
                )

    # def load_audio_file(
    #     self, filepath: str
    # ) -> Tuple[Optional[np.ndarray], Optional[int]]:  # cSpell:ignore ndarray
    #     """
    #     Load a WAV file and return audio data and sample rate.

    #     Args:
    #         filepath: Path to the WAV file

    #     Returns:
    #         Tuple of (audio_data, sample_rate)
    #     """
    #     try:
    #         sample_rate, audio_data = wavfile.read(filepath)

    #         # Convert to float32 and normalize
    #         if audio_data.dtype == np.int16:
    #             audio_data = audio_data.astype(np.float32) / 32768.0
    #         elif audio_data.dtype == np.int32:
    #             audio_data = audio_data.astype(np.float32) / 2147483648.0
    #         elif audio_data.dtype == np.uint8:
    #             audio_data = (audio_data.astype(np.float32) - 128) / 128.0
    #         elif audio_data.dtype == np.float32:
    #             # Already float32, but ensure it's in [-1, 1] range
    #             if np.max(np.abs(audio_data)) > 1.0:
    #                 audio_data = np.clip(audio_data, -1.0, 1.0)
    #         elif audio_data.dtype == np.float64:
    #             # Convert float64 to float32
    #             audio_data = audio_data.astype(np.float32)
    #         if np.max(np.abs(audio_data)) > 1.0:
    #             audio_data = np.clip(audio_data, -1.0, 1.0)
    #         else:
    #             self.logger.warning(
    #                 f"Unsupported audio format {audio_data.dtype} in {filepath}"
    #             )
    #         # Try to convert to float32 anyway
    #         audio_data = audio_data.astype(np.float32)

    #         # Ensure audio_data is contiguous in memory for better performance
    #         audio_data = np.ascontiguousarray(
    #             audio_data
    #         )  # cSpell:ignore ascontiguousarray

    #         return audio_data, sample_rate

    #     except FileNotFoundError:
    #         self.logger.error(f"File not found: {filepath}")
    #         return None, None
    #     except Exception as e:
    #         self.logger.error(f"Error loading {filepath}: {e}")
    #         return None, None

    def consolidate_audio_files(self, wave_files: dict) -> bool:
        """
        Recursively find all WAV files in directory and subdirectories.
        Args:
            root_directory: Path to the root directory to search
            wave_files: Dict of WAV file paths organized by Cohort and Session
                for example
                wave_files["Cohort2"]["Session 123"] = ["file1.wav", "file2.wav"]
        Returns:
            bool: Success indicator
        """
        try:
            for cohort in wave_files.keys():
                for session, files in wave_files[cohort].items():
                    self._combine_wave_file_list(files, cohort, session)
                    self.logger.info(f"Combined {len(files)} files in {session}")
            self.console.print("[green]Processing completed successfully![/green]")
            return True
        except Exception as e:
            self.logger.error(f"Error consolidating audio files: {e}")
            return False

    def load_noise_profile(self):
        # Load and compute noise spectrum from noise file
        try:
            noise_data = wavfile.read(self.npz_file)
            # convert to float and normalize
            noise_data_array = np.array(noise_data)
            converted_noise_data = noise_data_array.astype(float)
            if converted_noise_data.dtype == np.int16:
                return converted_noise_data.astype(np.float32) / 32_768.0
            elif converted_noise_data == np.int32:
                return converted_noise_data.astype(np.float32) / 2_147_483_648.0
        except Exception as e:
            raise ValueError(f"Error loading audio file: {e}")

    def compute_noise_spectrum(
        self,
        noise_data,
        nperseg=8192,  # cSpell:ignore nperseg
        overlap=0.75,
        window="hann",  # cSpell:ignore hann
        detrend="constant",  # cSpell:ignore detrend
    ):
        # using Welch's method
        noverlap = int(nperseg * overlap)  # cSpell:ignore noverlap

        frequencies, psd = signal.welch(
            noise_data,
            fs=self.sample_rate,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
            scaling="density",
        )

        return frequencies, psd

    def get_noise_floor_db(self, psd):
        return 10 * np.log10(psd + 1e-12)

    def get_US_band_noise(self, freq_min=15000, freq_max=120000):
        if self.noise_spectrum is None:
            raise ValueError("Compute noise spectrum first")

        freq_mask = (self.frequencies >= freq_min) & (self.frequencies <= freq_max)
        band_freqs = self.frequencies[freq_mask]  # cSpell:ignore freqs
        band_noise = self.noise_spectrum[freq_mask]

        stats = {
            "freq_range": (freq_min, freq_max),
            "mean_noise_linear": np.mean(band_noise),
            "mean_noise_db": 10 * np.log10(np.mean(band_noise) + 1e-12),
            "median_noise_db": 10 * np.log10(np.median(band_noise) + 1e-12),
            "std_noise_db": np.std(10 * np.log10(band_noise + 1e-12)),
            "frequencies": band_freqs,
            "noise_psd": band_noise,
        }
        return stats

    def save_noise_spectrum(self, frequencies, psd, sampling_rate: int = 256_000):
        """
        Save computed noise spectrum to file
        """
        np.savez(  # cSpell:ignore savez
            str(self.npz_file),
            frequencies=frequencies,
            noise_psd=psd,
            sampling_rate=sampling_rate,
        )
        print(f"Noise spectrum saved to {str(self.npz_file)}")

    def load_noise_spectrum(self, filepath):
        """
        Load previously computed noise spectrum

        Parameters:
        filepath (str): Input file path (NPZ format)
        """
        data = np.load(filepath)
        self.frequencies = data["frequencies"]
        self.noise_spectrum = data["noise_psd"]
        self.sample_rate = int(data["sampling_rate"])
        print(f"Noise spectrum loaded from {filepath}")

    def subtract_noise_from_signal(
        self, signal_psd, method="simple", alpha=2.0, beta=0.01
    ):
        """
        Perform spectral subtraction for noise reduction

        Parameters:
        signal_psd (array): Power spectral density of signal + noise
        method (str): 'simple' or 'wiener'
        alpha (float): Over-subtraction factor
        beta (float): Spectral floor factor

        Returns:
        array: Noise-reduced PSD
        """
        if self.noise_spectrum is None:
            raise ValueError("Compute noise spectrum first")

        if method == "simple":
            # Simple spectral subtraction
            clean_psd = signal_psd - alpha * self.noise_spectrum
            # Apply spectral floor
            clean_psd = np.maximum(clean_psd, beta * signal_psd)

        elif method == "wiener":
            # Wiener filter approach
            snr_est = signal_psd / (self.noise_spectrum + 1e-12)
            wiener_gain = snr_est / (1 + snr_est)
            clean_psd = wiener_gain * signal_psd

        else:
            raise ValueError("Method must be 'simple' or 'wiener'")

        return clean_psd

    def bandpass_filter(
        self, signal, sample_rate, low_freq=18000, high_freq=100000, order=5
    ):
        """
        Apply bandpass filter to signal

        Args:
            signal: Input signal
            sample_rate: Sample rate
            low_freq: Low cutoff frequency (Hz)
            high_freq: High cutoff frequency (Hz)
            order: Filter order

        Returns:
            Filtered signal
        """
        # cSpell:ignore nyquist
        # Check if frequencies are within nyquist limit
        nyquist = sample_rate / 2
        if high_freq >= nyquist:
            high_freq = nyquist * 0.95  # Set to 95% of Nyquist
            print(
                f"Warning: High frequency adjusted to {high_freq:.0f} Hz (Nyquist limit)"
            )

        if low_freq >= nyquist:
            print(
                f"Error: Low frequency {low_freq} Hz exceeds Nyquist frequency {nyquist} Hz"
            )
            return signal

        # Normalize frequencies
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist

        # Design Butterworth bandpass filter
        sos = butter(
            order,
            [low_norm, high_norm],
            btype="band",  # cSpell:ignore btype
            output="sos",
        )

        # Apply filter
        filtered_signal = sosfilt(sos, signal)
        return filtered_signal

        ##set up for current code above.

    def process_directory(
        self,
        input_dir: str | None = None,
        output_dir: str | None = None,
        noise_profile_path: str | None = None,
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

        # Find all WAV files
        wav_files = self.find_wave_files()

        if not wav_files:
            self.console.print("[red]No WAV files found in directory[/red]")
            return False

        # Display found files
        self.display_found_files(wav_files)

        # Consolidate the files
        # - 1. Protect against missing input
        if not self.root_path and not input_dir:
            raise ValueError("No input directory provided")
        if not self.root_path and input_dir is not None:
            self.root_path = Path(input_dir)
        # - 2. Create output directory if it doesn't exist
        self.combined_path.mkdir(parents=True, exist_ok=True)
        # - 3. Call the function that will loop and consolidate the audio files
        consolidated_success = self.consolidate_audio_files(wav_files)
        if not consolidated_success:
            return False

        # Apply filters if noise profile is provided
        # if noise_profile_path:
        #     self.filtered_path.mkdir(parents=True, exist_ok=True)
        #     filter_success = self.bandpass_filter(consolidated_path, noise_profile_path)
        #     if not filter_success:
        #         return False

        return True

    def display_found_files(self, wav_files: dict):
        """Display a table of found WAV files."""
        table = Table(title="Found WAV Files")
        table.add_column("Cohort", style="cyan", no_wrap=True)
        table.add_column("Session", style="magenta")
        table.add_column("Experiment Files", style="green")

        for cohort in wav_files.keys():
            for session in wav_files[cohort].keys():
                files = wav_files[cohort][session]
                table.add_row(cohort, session, str(len(files)))

        self.console.print(table)


if __name__ == "__main__":
    test_audio_processing = AudioProcessing(
        "/Users/djfurman/Downloads/amanda-data/Maternal_retrieval_USV"
    )
    result = test_audio_processing.process_directory()
    print(result)
