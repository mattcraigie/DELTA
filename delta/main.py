import argparse
import os
import yaml

from .data.preprocessing import preprocess_data

from .experiments.datascaling import run_datascaling_experiment
from .experiments.observables import run_observables_experiment
from .experiments.distances import run_distances_experiment
from .experiments.basic import run_basic_experiment


def load_config(config_path):
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main(config):
    # Data Preprocessing
    preprocess_data(config)

    # Experiments
    if config["experiments"]["run_datascaling_experiment"]:
        print("Running data scaling experiment...")
        run_datascaling_experiment(config)

    if config["experiments"]["run_observables_experiment"]:
        print("Running observables experiment...")
        run_observables_experiment(config)

    if config["experiments"]["run_distances_experiment"]:
        print("Running distances experiment...")
        run_distances_experiment(config)

    else:
        print("Running basic experiment...")
        run_basic_experiment(config)

    print("All experiments completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script for running experiments.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    # Load and parse the configuration
    config = load_config(args.config)

    # Run the main logic
    main(config)

