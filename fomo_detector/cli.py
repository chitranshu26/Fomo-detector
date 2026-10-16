import argparse

from fomo_detector.config import DATA_PATH, LOG_PATH, MODEL_PATH
from fomo_detector.simulate import run_simulation
from fomo_detector.train import train_model


def build_parser():
    parser = argparse.ArgumentParser(prog="fomo-detector")
    commands = parser.add_subparsers(dest="command")

    train_parser = commands.add_parser("train", help="train and save a classifier")
    train_parser.add_argument("--data", type=str, default=str(DATA_PATH))
    train_parser.add_argument("--model", type=str, default=str(MODEL_PATH))

    simulation_parser = commands.add_parser("simulate", help="run a synthetic simulation")
    simulation_parser.add_argument("--duration", type=float, default=300)
    simulation_parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    simulation_parser.add_argument("--log", type=str, default=str(LOG_PATH))
    return parser


def _run_interactive():
    parser = build_parser()
    while True:
        print("\nFOMO Detector")
        print("1. Train model")
        print("2. Run simulation")
        print("3. Exit")
        try:
            choice = input("Select an option: ").strip()
        except EOFError:
            return

        try:
            if choice == "1":
                train_model()
            elif choice == "2":
                duration_text = input("Simulation duration in seconds [300]: ").strip()
                duration = float(duration_text) if duration_text else 300
                run_simulation(duration=duration)
            elif choice == "3":
                return
            else:
                print("Choose 1, 2, or 3.")
        except (OSError, ValueError) as error:
            print(f"Error: {error}")


def main(argv=None):
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command is None:
        _run_interactive()
        return

    try:
        if arguments.command == "train":
            train_model(data_path=arguments.data, model_path=arguments.model)
        elif arguments.command == "simulate":
            run_simulation(
                duration=arguments.duration,
                model_path=arguments.model,
                log_path=arguments.log,
            )
    except (OSError, ValueError) as error:
        parser.error(str(error))