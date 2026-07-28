# Repository and Experiment Provenance

## Source snapshot

The clean repository was produced from Git commit:

`b2b3ffcd7e766831ce12a2cf38277ee152e5f8e9`

The later held-out snapshot was preserved as:

`ccc0e96e48b7311e1258371b83afc699bade976e`

The useful source changes from that snapshot were already present on `main`.
Only disposable schema-smoke result files were excluded.

## Checkpoints

Seven recovered schema-v3 checkpoint, metadata, validation, and trainer-state
files are distributed in the `schema-v3-checkpoints-20260727` GitHub Release.
The release includes `SHA256SUMS` for integrity verification.

## Interrupted evaluation

The July 2026 held-out evaluation campaign did not finish because its compute
machine permanently shut down. Partial results from that campaign must not be
treated as final experimental results.

The evaluation should be rerun from the beginning using fixed paired demand for
every controller. The rerun should preserve:

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
