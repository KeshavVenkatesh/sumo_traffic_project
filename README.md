# Map-Agnostic Traffic Signal Control in SUMO

Research code for training and evaluating a shared schema-v3 reinforcement-learning
traffic-signal controller across SUMO road networks with different intersection,
lane, and phase layouts.

## Repository status

- This clean source tree is based on commit `b2b3ffcd7e766831ce12a2cf38277ee152e5f8e9`.
- The July 2026 held-out evaluation was interrupted when its compute machine shut down.
- Incomplete evaluation outputs are not presented as final results.
- Model checkpoints are distributed through a GitHub Release.
- Generated maps, demand files, checkpoints, logs, and evaluation outputs are not committed.

## Setup

Install SUMO and ensure `sumo` is available on `PATH`. Then create a Python
environment:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements-dev.txt

Download and verify the schema-v3 checkpoints:

    mkdir -p models
    gh release download schema-v3-checkpoints-20260727 \
      --repo KeshavVenkatesh/sumo_traffic_project \
      --dir models
    (cd models && sha256sum -c SHA256SUMS)

Run the tests:

    python -m pytest -q

## Documentation

Training, demand generation, evaluation, and monitoring instructions are in
[docs/MAP_AGNOSTIC_TRAINING.md](docs/MAP_AGNOSTIC_TRAINING.md).

Repository and experiment provenance are recorded in
[docs/PROVENANCE.md](docs/PROVENANCE.md).
