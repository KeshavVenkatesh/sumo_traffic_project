# Repository and experiment provenance

## Normal-traffic baseline

The frozen ambulance input is
`models/map_agnostic_multiagent_v3_best.zip`. Checkpoints are distributed
separately through the `schema-v3-checkpoints-20260727` GitHub Release and must
be verified with that release's `SHA256SUMS`.

The completed July 30 schema-v3 campaign reported:

- 1,080 immutable fixed-demand runs;
- fixed-demand fairness passed;
- 4,222,800 learned-policy decisions;
- zero invalid learned-policy actions;
- all requested completion markers present.

The learned controller consistently outperformed Native SUMO. MaxPressure was
stronger on the newly generated Fresno and San Diego maps, so those maps remain
untouched final-test inputs unless they are explicitly converted to development
maps and replaced by new final tests.

## Ambulance schema v5

This ZIP is a reviewed source snapshot, not a trained ambulance result. Schema
v5 intentionally rejects prior ambulance checkpoints. A valid result requires:

- the ZIP SHA-256 reported with the release artifact;
- the exact frozen-base checkpoint SHA-256 stored in the v5 contract;
- checksummed network and fixed-demand files;
- a fresh smoke run and constrained training run;
- the `_best` checkpoint, contract, trainer state, and validation JSON;
- the paired five-way held-out evaluation JSON and its input manifest;
- SUMO, Python, PyTorch, and package versions.

Generated maps, demand banks, checkpoints, logs, and evaluation outputs are not
embedded in this source ZIP. Preserve them separately and do not modify Fresno
or San Diego after the held-out campaign begins.
