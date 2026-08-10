# Ambulance schema-v5 reviewed release

This source release supersedes the earlier schema-v4 package. It requires a
fresh emergency-override training run and deliberately rejects v4 checkpoints.
The validated normal-traffic input remains
`models/map_agnostic_multiagent_v3_best.zip`.

## Release-blocking fixes

- Ambulances scheduled exactly on a policy boundary are queued one SUMO step
  early with their original `depart` time. They are visible when the boundary
  decision is made without changing the exogenous demand schedule.
- A cosmetic `setColor()` failure can no longer convert a successful
  `vehicle.add()` request into a false insertion failure.
- Episode-end censored ambulances enter the terminal decision delta and reward,
  not just the final report.
- The 18 m receiving-lane gap is checked both when a phase is requested and
  immediately before green after yellow/all-red. If the exit fills during the
  transition, the controller remains all-red and retries.
- The unsafe hard-maximum-green forced-switch fallback is disabled in every
  ambulance ablation.
- SUMO automatic teleporting is disabled. Departures, arrivals, collisions,
  starting teleports, removals, failed insertions, and censored trips remain
  distinct lifecycle outcomes.
- Reroute jitter is generated independently per ambulance and reroute check, so
  a faster controller cannot change a later ambulance's randomness by finishing
  an earlier trip before it consumes another shared RNG draw.
- Ordinary wait is integrated at one-second resolution instead of using SUMO's
  rolling accumulated-wait memory. The promotion delay metric includes all
  departed ordinary vehicles, including those still active at the horizon.
- Fixed demand is bound to an audited passenger vType containing
  `jmIgnoreKeepClearTime="-1"`; old unsafe demand banks are rejected.
- Recovery imitation uses normalized MaxPressure, and conflicting ambulance
  phases are ranked by protection, urgency, load, current phase, and receiving
  space instead of numeric action order.
- Every worker receives a unique SUMO error log, and partially started worker
  groups are cleaned up after startup failures.
- Decision duration must divide exactly into the SUMO step and episode horizon;
  silent rounding and horizon overshoot are rejected.

## Promotion requirements

A checkpoint is promoted only when it satisfies the ordinary-delay and
throughput budgets, has zero lifecycle/safety/transition violations, is no
worse than frozen-base signaling in every paired validation scenario, and is
no worse than deterministic emergency preemption. Held-out evaluation still
uses the five paired routing/signaling ablations described in
`AMBULANCE_TRAINING.md`.

## Verification boundary

The source compiles and its dependency-free regression suite is included in
the ZIP. Neural checkpoint tests require PyTorch, and the real TraCI smoke run
must be executed on a machine with SUMO. Source review alone is not evidence
that a newly trained checkpoint meets the promotion gates.
