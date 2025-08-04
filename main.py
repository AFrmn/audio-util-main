import os
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from audio_processing import AudioProcessing


class FolderSelector:
    """GUI folder selection utility."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main window

    def select_input_folder(self) -> Optional[str]:
        """Open a folder selection dialog for input directory."""
        folder = filedialog.askdirectory(
            title="Select Input Directory (containing WAV files)", mustexist=True
        )
        return folder if folder else None

    def select_output_folder(self) -> Optional[str]:
        """Open a folder selection dialog for output directory."""
        folder = filedialog.askdirectory(
            title="Select Output Directory for consolidated files"
        )
        return folder if folder else None

    def select_final_folder(self) -> Optional[str]:
        """Open a folder selection dialog for processed files"""
        folder = filedialog.askdirectory(
            title="Select output directory for processed files."
        )
        return folder if folder else None

    def show_completion_message(self, output_dir: str):
        """Show completion message with output directory."""
        messagebox.showinfo(
            "Processing Complete",
            f"Processing completed successfully!\n\nOutput files saved to:\n{output_dir}",
        )

    def show_error_message(self, message: str):
        """Show error message."""
        messagebox.showerror("Error", message)

    def select_noise_file(self) -> Optional[str]:
        """Open a file selection dialog for multiple noise files."""
        noise_file = filedialog.askopenfilename(
            title="Select input directory for the noise file", defaultextension=".wav"
        )
        return noise_file if noise_file else None


# Initialize Typer app
app = typer.Typer(
    help="WAV File Consolidator and Filter Tool for Scientific Experiments",
    rich_markup_mode="rich",
)

console = Console()


@app.command()
def process(
    input_dir: Annotated[
        Optional[str], typer.Argument(help="Input directory containing WAV files")
    ] = None,
    output_dir: Annotated[
        Optional[str], typer.Argument(help="Output directory for processed files")
    ] = None,
    gui: Annotated[
        bool, typer.Option("--gui", help="Use GUI folder selection")
    ] = False,
    noise_profile: Annotated[
        Optional[str], typer.Option(help="Path to noise profile file")
    ] = None,
):
    """
    Process WAV files: consolidate and apply filters.

    Available presets: scientific, noise_reduction, vocal_enhancement, adaptive_noise_reduction
    """

    # Display welcome message
    console.print(
        Panel.fit(
            "[bold blue]🎵 WAV File Consolidator & Filter Tool[/bold blue]\n"
            "[dim]For Scientific Experiments[/dim]",
            border_style="blue",
        )
    )

    processor = AudioProcessing(input_dir)

    # Handle GUI mode or missing arguments
    if gui or not input_dir or not output_dir:
        folder_selector = FolderSelector()

        if not input_dir:
            console.print(
                "\n[yellow]Please select the input directory containing WAV files...[/yellow]"
            )
            input_dir = folder_selector.select_input_folder()
            if not input_dir:
                console.print("[red]No input directory selected. Exiting.[/red]")
                return

        if not output_dir:
            console.print(
                "\n[yellow]Please select the output directory for processed files...[/yellow]"
            )
            output_dir = folder_selector.select_output_folder()
            if not output_dir:
                console.print("[red]No output directory selected. Exiting.[/red]")
                return

        # Handle noise profile selection
        if gui and not noise_profile:
            if Confirm.ask(
                "\n[yellow]Do you have a noise profile for better noise reduction?[/yellow]",
                default=False,
            ):
                console.print(
                    "[yellow]Please select the noise profile file...[/yellow]"
                )
                noise_profile = folder_selector.select_noise_file()
                if noise_profile:
                    console.print(
                        f"[green]Selected noise profile: {os.path.basename(noise_profile)}[/green]"
                    )
                else:
                    console.print(
                        "[yellow]No noise profile selected. Proceeding without noise profiling.[/yellow]"
                    )

    # Display processing parameters
    params_table = Table(title="Processing Parameters")
    params_table.add_column("Parameter", style="cyan")
    params_table.add_column("Value", style="green")

    params_table.add_row("Input Directory", input_dir)
    params_table.add_row("Output Directory", output_dir)
    params_table.add_row("Noise Profile", noise_profile if noise_profile else "None")

    console.print(params_table)

    # Confirm processing
    if not Confirm.ask("\n[bold]Proceed with processing?[/bold]", default=True):
        console.print("[yellow]Processing cancelled.[/yellow]")
        return

    # Process the directory
    if noise_profile:
        success = processor.process_directory(input_dir, output_dir, noise_profile)
    else:
        success = processor.process_directory(input_dir, output_dir, "")

    if success:
        console.print("\n[green]✅ Processing completed successfully![/green]")
        console.print(f"[blue]📁 Output files saved to: {output_dir}[/blue]")

        # Show GUI completion message if GUI was used
        if gui:
            folder_selector = FolderSelector()
            folder_selector.show_completion_message(output_dir)
    else:
        console.print("\n[red]❌ Processing failed. Check the logs for details.[/red]")
        if gui:
            folder_selector = FolderSelector()
            folder_selector.show_error_message(
                "Processing failed. Check the console for details."
            )


@app.command()
def interactive():
    """
    Run in interactive mode with prompts for all parameters.
    """
    console.print(
        Panel.fit(
            "[bold green]🎛️ Interactive Mode[/bold green]\n"
            "[dim]Step-by-step configuration[/dim]",
            border_style="green",
        )
    )

    # Initialize noise_profile variable
    noise_profile = None

    # Get input directory
    use_gui = Confirm.ask("Use GUI for folder selection?", default=True)

    if use_gui:
        folder_selector = FolderSelector()
        console.print("\n[yellow]Select input directory...[/yellow]")
        input_dir = folder_selector.select_input_folder()
        if not input_dir:
            console.print("[red]No input directory selected. Exiting.[/red]")
            return

        console.print("\n[yellow]Select output directory...[/yellow]")
        output_dir = folder_selector.select_output_folder()
        if not output_dir:
            console.print("[red]No output directory selected. Exiting.[/red]")
            return

        # Handle noise profile selection
        if Confirm.ask(
            "\n[yellow]Do you have a noise profile for better noise reduction?[/yellow]",
            default=False,
        ):
            console.print(
                "[yellow]Please select the noise profile directory...[/yellow]"
            )
            noise_profile = folder_selector.select_noise_file()
            if noise_profile:
                console.print(
                    f"[green]Selected noise profile: {os.path.basename(noise_profile)}[/green]"
                )
            else:
                console.print(
                    "[yellow]No noise profile selected. Proceeding without noise profiling.[/yellow]"
                )

    else:
        input_dir = Prompt.ask("Enter input directory path")
        output_dir = Prompt.ask("Enter output directory path")

        if Confirm.ask("Do you have a noise profile?", default=False):
            noise_profile = Prompt.ask("Enter noise profile path")

    # Display processing parameters
    params_table = Table(title="Processing Parameters")
    params_table.add_column("Parameter", style="cyan")
    params_table.add_column("Value", style="green")

    params_table.add_row("Input Directory", input_dir)
    params_table.add_row("Output Directory", output_dir)
    params_table.add_row("Noise Profile", noise_profile if noise_profile else None)

    console.print(params_table)

    # Confirm processing
    if not Confirm.ask("\n[bold]Proceed with processing?[/bold]", default=True):
        console.print("[yellow]Processing cancelled.[/yellow]")
        return

    # Process
    processor = AudioProcessing(input_dir)
    if noise_profile:
        success = processor.process_directory(input_dir, output_dir, noise_profile)
    else:
        success = processor.process_directory(input_dir, output_dir, "")

    if success:
        console.print("\n[green]✅ Processing completed successfully![/green]")
        console.print(f"[blue]📁 Output files saved to: {output_dir}[/blue]")

    if use_gui:
        folder_selector.show_completion_message(output_dir)  # type: ignore
    else:
        console.print("\n[red]❌ Processing failed. Check the logs for details.[/red]")
    if use_gui:
        folder_selector = FolderSelector()
        folder_selector.show_error_message(
            "Processing failed. Check the console for details."
        )


if __name__ == "__main__":
    app()
