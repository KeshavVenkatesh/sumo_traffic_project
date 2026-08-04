# Map-agnostic multi-agent schema-v3 result

This directory records the completed evaluation of the frozen non-ambulance
checkpoint `models/map_agnostic_multiagent_v3_best.zip`. The learned policy is
called `all_model` in the evaluation output because one shared model controls
all compatible traffic-light intersections simultaneously.

## Reproducibility identity

- Source commit: `39408d577806b8136c6b0290394ecd4c532b2f47`
- Checkpoint SHA-256:
  `be832797883d589441fd27d52b87884320a624af024eab0dac4ed8d9860f0706`
- Maps: Fremont, Santa Clara, Fresno, and San Diego
- Demand rates: 6, 12, and 18 trips per lane-km per hour
- Seeds: 1001 through 1030
- Controllers: native SUMO, normalized MaxPressure, and `all_model`
- Episode length: 1,200 simulated seconds
- Fixed route schedules: 360
- Total evaluations: 1,080

The completion gate reported:

```text
FULL_RUNS=1080
FULL_FIXED_DEMAND_FAIRNESS=passed
FULL_POLICY_INVALID_ACTIONS=0
IMMUTABLE_COMPREHENSIVE_EVALUATION_PASSED
```

## Learned controller versus native SUMO

Pooled across the full campaign, the learned controller produced the following
improvements. Positive traffic throughput/speed values mean higher is better;
the congestion values are reported as reductions.

| Metric | Improvement |
| --- | ---: |
| Arrivals | 7.42% higher |
| Mean speed | 8.42% higher |
| Mean queue | 22.58% lower |
| Maximum queue | 29.38% lower |
| Mean wait | 19.64% lower |
| Maximum wait | 18.63% lower |
| Recovery interventions | 45.15% fewer |

Arrivals, speed, mean queue, maximum queue, and recovery interventions were
clear improvements in all 12 map/rate conditions after Holm correction.
Waiting-time results were not uniformly positive across conditions, so the
pooled waiting improvements should not be interpreted as universal wins.

On the held-out Fresno and San Diego maps, the learned controller achieved:

- 1.57% to 6.82% more arrivals;
- 3.79% to 11.93% higher mean speed;
- 10.62% to 28.64% lower mean queues;
- 11.75% to 39.98% lower maximum queues; and
- 30.57% to 39.67% fewer recovery interventions.

Normalized MaxPressure was stronger than the learned controller on the new
Fresno and San Diego maps. The result therefore establishes improvement over
native SUMO, not superiority over every adaptive controller.

`evaluation_manifest.json` contains the machine-readable scope and headline
results. Training, fixed-demand generation, evaluation, and analysis code are
already maintained in the repository root and documented in
`docs/MAP_AGNOSTIC_TRAINING.md`; they are not duplicated here.
