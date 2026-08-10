# Map-Agnostic Traffic Signal Control in SUMO

Research code for training and evaluating a shared schema-v3 reinforcement-learning
traffic-signal controller across SUMO road networks with different intersection,
lane, and phase layouts.

## Repository status

- The 1,080-run schema-v3 held-out traffic evaluation completed with fixed
  demand, zero invalid policy actions, and immutable output. The learned base
  controller consistently beat Native SUMO, while MaxPressure remained
  stronger on the newly generated Fresno and San Diego maps.
- Ambulance schema v5 is a reviewed source release around that frozen v3 base.
  It must be trained fresh and promoted through its constrained validation
  gates; no unvalidated emergency weights are bundled here.
- Model checkpoints are distributed separately through a GitHub Release.
- Generated maps, demand files, checkpoints, logs, and evaluation outputs are
  not committed.

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

Ambulance routing, emergency signal-priority training, constrained checkpoint
selection, and the paired five-way evaluation are documented in
[docs/AMBULANCE_TRAINING.md](docs/AMBULANCE_TRAINING.md).

For the reviewed ambulance package, begin with
[AMBULANCE_READ_ME_FIRST.md](AMBULANCE_READ_ME_FIRST.md).

Repository and experiment provenance are recorded in
[docs/PROVENANCE.md](docs/PROVENANCE.md).
