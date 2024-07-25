"""Run the examples and generate the documentation."""

from pathlib import Path

from click.testing import CliRunner

from wave_packet_dynamics.cli import animate
from wave_packet_dynamics.cli import run


runner = CliRunner()
for input_file in Path("examples/").glob("*.toml"):
    try:
        runner.invoke(run, [str(input_file)])
    except FileExistsError:
        continue
    save_directory = input_file.parent / input_file.stem
    if not (save_directory / "animation.gif").exists():
        runner.invoke(animate, [str(save_directory)])
