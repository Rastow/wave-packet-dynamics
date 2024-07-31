"""Command line interface."""

import time
import tomllib

from functools import partial
from pathlib import Path

import click

import wave_packet_dynamics as wpd


@click.group()
@click.version_option(version=wpd.__version__)
def cli() -> None:
    """\b
      _      _____  ___
     | | /| / / _ \\/ _ \\
     | |/ |/ / ___/ // /
     |__/|__/_/  /____/

    Simulation of quantum dynamics.
    """  # noqa: D205, D301, D400


@click.command()
@click.option(
    "--save-step",
    type=int,
    help="Save simulation data every n-th step.",
    default=10,
    show_default=True,
)
@click.option(
    "--save-dir",
    type=click.Path(exists=True, path_type=Path),
    help="Output directory.",
    default=None,
    show_default="same as INPUT_FILE",
)
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
def run(save_step: int, save_dir: Path, input_file: Path) -> None:
    """Run the simulation using an INPUT_FILE."""
    # load the configuration file
    with input_file.open("rb") as file:
        config = tomllib.load(file)

    # create and run the simulation using the configuration
    simulation = wpd.Simulation(
        grid=wpd.Grid(
            bounds=(config["grid"]["min"], config["grid"]["max"]), points=config["grid"]["points"]
        ),
        particle=getattr(wpd.particles, str.capitalize(config["particle"]))
        if isinstance(config["particle"], str)
        else wpd.particles.Particle(**config["particle"]),
        wave_function=partial(
            getattr(wpd.wave_function, str.lower(config["wave_function"].pop("type"))),
            **config["wave_function"],
        ),
        potential=partial(
            getattr(wpd.potential, str.lower(config["potential"].pop("type"))),
            **config["potential"],
        ),
    )
    click.echo("Running simulation...")
    start = time.time()
    simulation(
        time_interval=config["time_interval"],
        total_time_steps=config["total_time_steps"],
        save_step=save_step,
        save_directory=input_file.parent / input_file.stem if save_dir is None else save_dir,
    )
    end = time.time()
    click.echo(f"Simulation completed after {end - start:.2f} seconds.")


@click.command()
@click.argument("output_dir", type=click.Path(exists=True, path_type=Path))
def animate(output_dir: Path) -> None:
    """Animate the simulation results of data stored in OUTPUT_DIR."""
    animation = wpd.Animation(Path(click.format_filename(output_dir)))
    click.echo("Creating animation...")
    animation.animation.save(filename=output_dir / "animation.gif", writer="pillow", dpi=200)
    click.echo("Animation created.")


@click.command()
@click.argument("output_dir", type=click.Path(exists=True, path_type=Path))
def plot(output_dir: Path) -> None:
    """Plot expectation values of simulation stored in OUTPUT_DIR."""
    plot = wpd.Plot(Path(click.format_filename(output_dir)))
    click.echo("Plotting expectation values...")
    plot.fig.savefig(output_dir / "plot.png", dpi=200)
    click.echo("Plot created.")


cli.add_command(run)
cli.add_command(animate)
cli.add_command(plot)
