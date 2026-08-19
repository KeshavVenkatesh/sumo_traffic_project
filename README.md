# Map-Agnostic Traffic Signal Control in SUMO

Research code for training and evaluating a shared schema-v3 reinforcement-learning
traffic-signal controller across SUMO road networks with different intersection,
lane, and phase layouts.

## Repository status

- The cleaned schema-v3 source was merged at commit
  `39408d577806b8136c6b0290394ecd4c532b2f47`.
- The frozen best checkpoint completed an immutable 1,080-run evaluation on
  July 30, 2026, using paired fixed demand on Fremont, Santa Clara, Fresno,
  and San Diego.
- The learned controller beat native SUMO in all 12 map/rate conditions for
  arrivals, speed, mean queue, max queue, and recovery interventions after
  Holm correction. Waiting-time results were not uniformly positive.
- Model checkpoints are distributed through a GitHub Release.
- Generated maps, demand files, checkpoints, logs, and raw evaluation outputs
  are not committed.

The verified evaluation scope, checkpoint hash, and result summary are in
[results/map_agnostic_multiagent_v3](results/map_agnostic_multiagent_v3).

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
 
