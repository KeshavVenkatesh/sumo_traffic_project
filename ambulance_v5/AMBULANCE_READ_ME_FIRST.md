# Ambulance v5: read this first

This ZIP contains the reviewed schema-v5 source, tests, demand generator,
training pipeline, constrained checkpoint selector, and paired evaluator. It
does **not** contain a newly trained emergency checkpoint: that checkpoint must
be trained and promoted on the SUMO host. The required frozen input is your
existing `models/map_agnostic_multiagent_v3_best.zip`.

Do not resume any single-intersection, v4, or pre-arrival-fix ambulance
checkpoint. Schema v5 rejects those files by design.

## Quick start

1. Extract this ZIP into a new directory; do not overwrite an evaluation that
   is currently running.
2. Make `generated_map_corpus/` and
   `models/map_agnostic_multiagent_v3_best.zip` available in that directory
   (copy or symlink them from the existing project).
3. Activate the SUMO/PyTorch environment and set `SUMO_HOME`/`PYTHONPATH`.
4. Run:

   ```bash
   python -m compileall -q .
   python -m pytest -q
   ```

5. Follow `docs/AMBULANCE_TRAINING.md` from demand generation through the
   one-map smoke test. Generate fresh `*_v5` demand banks; old banks without
   the audited fixed-demand vehicle type are intentionally rejected.

The smoke test must complete a full 3,600-second episode and report at least
one real arrival, positive finite `response_s`, and zero failed/censored trips.
Only a checkpoint ending in `_best` and accompanied by its validation JSON has
passed the promotion gates.

See `docs/AMBULANCE_V5_RELEASE_NOTES.md` for the defects fixed in this review.
