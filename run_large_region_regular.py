import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_SUMO_HOME = (
    "/Library/Frameworks/EclipseSUMO.framework/"
    "Versions/1.26.0/EclipseSUMO/share/sumo"
)

os.environ.setdefault("SUMO_HOME", DEFAULT_SUMO_HOME)

BASE_DIR = Path(__file__).resolve().parent

SUMO_BIN_DIR = Path("/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/bin")
SUMO_GUI = SUMO_BIN_DIR / "sumo-gui"
SUMO = SUMO_BIN_DIR / "sumo"

if not SUMO_GUI.exists():
    SUMO_GUI = "sumo-gui"

if not SUMO.exists():
    SUMO = "sumo"


def ensure_xquartz():
    if sys.platform != "darwin":
        return

    os.environ.setdefault("DISPLAY", ":0")

    try:
        subprocess.run(
            ["open", "-a", "XQuartz"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        time.sleep(2)
    except Exception as e:
        print(f"Warning: could not open XQuartz automatically: {e}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--net",
        default="large_region.net.xml",
        help="SUMO network file.",
    )

    parser.add_argument(
        "--routes",
        default="background_large_region.rou.xml",
        help="SUMO route file.",
    )

    parser.add_argument(
        "--step-length",
        type=float,
        default=0.5,
        help="SUMO step length.",
    )

    parser.add_argument(
        "--end",
        type=float,
        default=7200,
        help="Simulation end time.",
    )

    parser.add_argument(
        "--max-vehicles",
        type=int,
        default=8000,
        help="Maximum active vehicles.",
    )

    parser.add_argument(
        "--max-depart-delay",
        type=int,
        default=120,
        help="Maximum departure delay.",
    )

    parser.add_argument(
        "--time-to-teleport",
        type=int,
        default=300,
        help="Teleport stuck vehicles after this many seconds.",
    )

    parser.add_argument(
        "--nogui",
        action="store_true",
        help="Run headless SUMO instead of sumo-gui.",
    )

    args = parser.parse_args()

    net_file = BASE_DIR / args.net
    route_file = BASE_DIR / args.routes

    if not net_file.exists():
        raise FileNotFoundError(f"Missing network file: {net_file}")

    if not route_file.exists():
        raise FileNotFoundError(f"Missing route file: {route_file}")

    if not args.nogui:
        ensure_xquartz()

    binary = str(SUMO) if args.nogui else str(SUMO_GUI)

    sumo_cmd = [
        binary,
        "-n", str(net_file),
        "-r", str(route_file),
        "--start",
        "--step-length", str(args.step_length),
        "--end", str(args.end),
        "--max-num-vehicles", str(args.max_vehicles),
        "--max-depart-delay", str(args.max_depart_delay),
        "--time-to-teleport", str(args.time_to_teleport),
        "--no-warnings", "true",
        "--no-step-log", "true",
    ]

    print("Starting regular large-region SUMO simulation:")
    print(" ".join(sumo_cmd))

    subprocess.run(sumo_cmd, check=True)


if __name__ == "__main__":
    main()
