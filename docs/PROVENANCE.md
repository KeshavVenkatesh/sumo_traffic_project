# Repository and Experiment Provenance

## Source snapshot

The cleaned source was first published at Git commit:

`b2b3ffcd7e766831ce12a2cf38277ee152e5f8e9`

The exact source tree used for the completed comprehensive evaluation was:

`39408d577806b8136c6b0290394ecd4c532b2f47`

The later held-out snapshot was preserved as:

`ccc0e96e48b7311e1258371b83afc699bade976e`

The useful source changes from that snapshot were already present on `main`.
Only disposable schema-smoke result files were excluded.

## Checkpoints

Seven recovered schema-v3 checkpoint, metadata, validation, and trainer-state
files are distributed in the `schema-v3-checkpoints-20260727` GitHub Release.
The release includes `SHA256SUMS` for integrity verification.

The deployment checkpoint selected by held-out validation and used in the
completed evaluation was `models/map_agnostic_multiagent_v3_best.zip`, with
SHA-256 digest:

`be832797883d589441fd27d52b87884320a624af024eab0dac4ed8d9860f0706`

## Completed comprehensive evaluation

An immutable comprehensive evaluation completed on July 30, 2026. It covered
Fremont, Santa Clara, Fresno, and San Diego; three demand rates; 30 paired
seeds; and native SUMO, normalized MaxPressure, and the learned schema-v3
controller. All 1,080 runs completed, fixed-demand fairness checks passed, and
the learned policy produced zero invalid actions.

The concise verified record is stored in
`results/map_agnostic_multiagent_v3/`. Raw generated routes, per-run logs, and
the 1,080 raw result files remain outside Git because they are generated
campaign artifacts.

## Earlier interrupted evaluation

An earlier July 2026 held-out campaign did not finish because its compute
machine permanently shut down. Its partial results remain superseded and must
not be combined with the completed July 30 campaign.

Any future rerun should use fixed paired demand for every controller and
preserve:

- Exact network and route files
- SHA-256 hashes of evaluation inputs
- Per-seed results
- Aggregate results
- SUMO, Python, PyTorch, and package versions
- The exact source commit and checkpoint hashes

The exact generated Fresno and San Diego files from the interrupted machine were
not committed, so the replacement campaign must regenerate and then preserve its
own immutable evaluation inputs.

## Historical repository

The original large repository and its branches should remain available as
`sumo_traffic_project_archive_20260727` until the cleaned repository and recovered
checkpoints have been independently verified.
