# Audio Util

This is an audio utility designed to take a session's worth of experimental data, consolidate the audio, and apply filters to it in order to clean it up for other analysis to take over.

The utility is coded as an interactive application allowing you to get real time feedaback of what it is working on doing as well as what inputs are necessary.

This application leverages [`numpy`](https://numpy.org/), [`scipy`](https://scipy.org/), and [`typer`](https://typer.tiangolo.com/) libraries. This was written using the [Claude.ai](https://claude.ai) AI utility.

## Usage

This repository manages its dependencies using the excellent [`uv`](https://docs.astral.sh/uv/) project from Astral. Installation instructions follow. The application expects that you can navigate to your audio files in a standard file explorer. Note that if the files are on your hard drive this will run much faster than if they are on an external drive or a cloud sync'd drive.

The application is designed to have you specify both an input and output directory and your files will remain unedited in the input directory, so all actions are non-destructive.

## Installation

Ensure that Python is installed on your machine. Install Python if not already done.

### Install uv

If on a Mac, install [homebrew](https://brew.sh/) if not already done, and run run `brew install uv`.
If on a Windows machine, install with pipx by running `pipx install uv`.

### Install this application

1. Clone this repo
1. Open a terminal an `cd` into this repo
1. Run `uv sync` to pull the dependencies

## Running the Application

1. Open a terminal an `cd` into this repo
1. Run `uv run main.py` from the terminal
1. Follow the interactive instructions
