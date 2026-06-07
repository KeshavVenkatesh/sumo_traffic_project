import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_SUMO_HOME = (
    "/Library/Frameworks/EclipseSUMO.framework/"
    "Versions/1.26.0/EclipseSUMO/share/sumo"
)

os.environ.setdefault("SUMO_HOME", DEFAULT_SUMO_HOME)

BASE_DIR = Path(__file__).resolve().parent

SUMO_BIN_DIR = Path("/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/bin")
NETCONVERT = SUMO_BIN_DIR / "netconvert"
SUMO_GUI = SUMO_BIN_DIR / "sumo-gui"

RANDOM_TRIPS = Path(os.environ["SUMO_HOME"]) / "tools" / "randomTrips.py"


def run_cmd(cmd):
    print("\nRunning:")
    print(" ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def patch_car_vtype(route_file):
    path = Path(route_file)
    text = path.read_text()

    # Remove existing car definition if randomTrips created one.
    text = re.sub(r'\s*<vType id="car"[\s\S]*?/>', '', text)

    car_vtype = '''    <vType id="car"
           accel="2.6"
           decel="4.5"
           emergencyDecel="9.0"
           maxSpeed="13.9"
           sigma="0.5"
           lcCooperative="1.0"
           lcStrategic="1.0"
           lcSpeedGain="1.0"
           jmIgnoreKeepClearTime="-1"
           jmDriveAfterYellowTime="-1"
           jmDriveAfterRedTime="-1"/>
'''

    text = re.sub(
        r'(<routes[^>]*>)',
        r'\1\n' + car_vtype,
        text,
        count=1,
    )

    path.write_text(text)
    print(f"\nPatched vehicle type in: {path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--osm",
        default="large_region.osm",
        help="Input OSM file.",
    )

    parser.add_argument(
        "--net",
        default="large_region.net.xml",
        help="Output SUMO network file.",
    )

    parser.add_argument(
        "--routes",
        default="background_large_region.rou.xml",
        help="Output route file.",
    )

    parser.add_argument(
        "--begin",
        type=float,
        default=0,
        help="Traffic begin time.",
    )

    parser.add_argument(
        "--end",
        type=float,
        default=7200,
        help="Traffic end time.",
    )

    parser.add_argument(
        "--period",
        type=float,
        default=0.35,
        help="randomTrips period. Smaller means more traffic.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    osm_file = BASE_DIR / args.osm
    net_file = BASE_DIR / args.net
    route_file = BASE_DIR / args.routes

    if not osm_file.exists():
        raise FileNotFoundError(f"Could not find OSM file: {osm_file}")

    if not NETCONVERT.exists():
        raise FileNotFoundError(f"Could not find netconvert: {NETCONVERT}")

    if not RANDOM_TRIPS.exists():
        raise FileNotFoundError(f"Could not find randomTrips.py: {RANDOM_TRIPS}")

    print("\n=== Step 1: Convert OSM to SUMO network ===")

    run_cmd([
        str(NETCONVERT),
        "--osm-files", str(osm_file),
        "-o", str(net_file),
        "--geometry.remove",
        "--ramps.guess",
        "--junctions.join",
        "--tls.guess",
        "--tls.join",
        "--tls.discard-simple",
        "--remove-edges.isolated",
        "--no-turnarounds",
    ])

    print("\n=== Step 2: Generate regular background traffic ===")

    run_cmd([
        sys.executable,
        str(RANDOM_TRIPS),
        "-n", str(net_file),
        "-r", str(route_file),
        "--no-validate",
        "-b", str(args.begin),
        "-e", str(args.end),
        "-p", str(args.period),
        "--seed", str(args.seed),
        "--prefix", "large_car_",
        "-t", 'type="car" departLane="best" departPos="free" departSpeed="max"',
    ])

    print("\n=== Step 3: Patch route file vehicle type ===")
    patch_car_vtype(route_file)

    print("\nSetup complete.")
    print(f"Network file: {net_file}")
    print(f"Route file:   {route_file}")

    print("\nTo run the regular simulation:")
    print("python3 run_large_region_regular.py")


if __name__ == "__main__":
    main()
