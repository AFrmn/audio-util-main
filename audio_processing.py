import logging
import os
import glob
import wave
from pathlib import Path
from typing import Optional

import scipy.io.wavfile as wavfile  # cSpell:ignore wavfile
from rich.console import Console
import noisereduce as nr
import numpy
import librosa

from rich.table import Table
from scipy.signal import butter, filtfilt  # cSpell:ignore filtfilt


class AudioProcessing:
    def __init__(self, root_directory: Optional[str] = None):
        if root_directory is None:
            raise ValueError("root_directory cannot be None")
        self.root_path = Path(root_directory)
        self.combined_path = self.root_path / Path("combined")
        self.filtered_path = self.root_path / Path("filtered")
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

    def _process_audio_files(self, wave_files: dict) -> None:
        for cohort in wave_files.keys():
        for session in wave_files[cohort].keys():
            combined_file = self.combined_path / cohort / f"{session}.wav"
        if combined_file.exists():
            output_file = self.filtered_path / cohort / f"{session}_filtered.wav"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            self._process_audio_file(
                str(combined_file),
                self.noise_profile_path,
                str(output_file)
            )

    def _process_audio_file(self, audio_path, noise_path, save_path) -> None:
        try:
            print(f"Loading: {audio_path}")
    
        # Load audio and noise files
        audio, sr = librosa.load(audio_path, sr=None)
        noise, _ = librosa.load(noise_path, sr=sr)
        print(f"Successfully loaded {audio_path}, shape: {audio.shape}, sample rate: {sr}")
    
    # Apply high-pass filter
        nyquist = sr / 2
        high_cutoff = 10_000 / nyquist
        b, a = butter(4, high_cutoff, btype="highpass", analog=False, output="ba")
        audio_filtered = filtfilt(b, a, audio)
        noise_filtered = filtfilt(b, a, noise)

    # Apply noise reduction
        cleaned_audio = nr.reduce_noise(
            y=audio_filtered,
            sr=sr,
            y_noise=noise_filtered,
            prop_decrease=0.7,
            chunk_size=512,
        )
    
    # Save the processed audio
        wavfile.write(save_path, sr, cleaned_audio.astype(audio.dtype))
        self.logger.info(f"Processed noise reduction and saved to {save_path}")
    
        except FileNotFoundError as e:
        self.logger.error(f"File not found: {audio_path} or {noise_path}")
        except Exception as e:
        self.logger.error(f"Error processing {audio_path}: {e}")
        
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
        if noise_profile_path:
            self.filtered_path.mkdir(parents=True, exist_ok=True)
            self.noise_profile_path = noise_profile_path
            self._process_audio_files(wave_files=wav_files)

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