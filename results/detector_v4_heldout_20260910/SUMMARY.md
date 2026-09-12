# Detector-v4 held-out evaluation

Checkpoint selection was fixed before testing.
All simulations used libsumo 1.26.0 and the documented 1,200-second final-evaluation horizon.
The 11,520 paired comparison rows come from 5,040 unique simulations: sensor-independent baselines ran once per map/rate/seed and were reused across sensor comparisons.

All percentage changes below are oriented so positive is better. Average rates within each map, then weight the eight maps equally.
The bootstrap interval resamples the eight map summaries (10,000 replicates). It does not establish performance on every future city.

| Sensors | Metric | Baseline | Mean improvement % | 95% map bootstrap interval |
|---|---|---|---:|---|
| loops | total_arrived | native_sumo | 12.43 | [7.92, 17.73] |
| loops | total_arrived | max_pressure | 9.46 | [4.59, 16.57] |
| loops | total_arrived | schema_v3 | 14.21 | [6.23, 26.84] |
| loops | mean_avg_speed_mps | native_sumo | 21.42 | [11.27, 36.75] |
| loops | mean_avg_speed_mps | max_pressure | 6.76 | [-1.02, 19.54] |
| loops | mean_avg_speed_mps | schema_v3 | 18.22 | [4.71, 41.93] |
| loops | mean_global_queue | native_sumo | 34.00 | [24.90, 45.28] |
| loops | mean_global_queue | max_pressure | 17.75 | [6.70, 33.37] |
| loops | mean_global_queue | schema_v3 | 25.09 | [12.89, 41.64] |
| loops | mean_global_wait | native_sumo | 27.51 | [3.82, 51.42] |
| loops | mean_global_wait | max_pressure | 28.80 | [9.34, 49.84] |
| loops | mean_global_wait | schema_v3 | 42.07 | [21.73, 61.62] |
| camera | total_arrived | native_sumo | 12.09 | [7.46, 17.56] |
| camera | total_arrived | max_pressure | 9.11 | [4.19, 16.11] |
| camera | total_arrived | schema_v3 | 13.83 | [5.82, 26.26] |
| camera | mean_avg_speed_mps | native_sumo | 20.17 | [10.40, 34.81] |
| camera | mean_avg_speed_mps | max_pressure | 5.66 | [-1.94, 17.80] |
| camera | mean_avg_speed_mps | schema_v3 | 16.93 | [3.77, 39.69] |
| camera | mean_global_queue | native_sumo | 32.83 | [23.64, 44.02] |
| camera | mean_global_queue | max_pressure | 16.27 | [4.76, 31.99] |
| camera | mean_global_queue | schema_v3 | 23.77 | [11.02, 40.51] |
| camera | mean_global_wait | native_sumo | 25.20 | [0.04, 49.76] |
| camera | mean_global_wait | max_pressure | 26.38 | [5.02, 48.10] |
| camera | mean_global_wait | schema_v3 | 40.02 | [17.60, 60.58] |
| mixed | total_arrived | native_sumo | 12.49 | [7.88, 17.85] |
| mixed | total_arrived | max_pressure | 9.52 | [4.50, 16.61] |
| mixed | total_arrived | schema_v3 | 14.27 | [6.17, 26.91] |
| mixed | mean_avg_speed_mps | native_sumo | 21.14 | [11.10, 36.18] |
| mixed | mean_avg_speed_mps | max_pressure | 6.53 | [-1.21, 19.04] |
| mixed | mean_avg_speed_mps | schema_v3 | 17.93 | [4.43, 41.34] |
| mixed | mean_global_queue | native_sumo | 33.51 | [24.37, 44.72] |
| mixed | mean_global_queue | max_pressure | 17.07 | [5.75, 32.83] |
| mixed | mean_global_queue | schema_v3 | 24.48 | [11.87, 41.33] |
| mixed | mean_global_wait | native_sumo | 25.87 | [1.05, 50.23] |
| mixed | mean_global_wait | max_pressure | 27.00 | [6.36, 48.51] |
| mixed | mean_global_wait | schema_v3 | 40.58 | [18.84, 60.84] |
| mixed_corrupted | total_arrived | native_sumo | 0.83 | [-4.81, 6.41] |
| mixed_corrupted | total_arrived | max_pressure | -2.09 | [-6.27, 1.01] |
| mixed_corrupted | total_arrived | schema_v3 | 1.83 | [-4.26, 7.71] |
| mixed_corrupted | mean_avg_speed_mps | native_sumo | 2.32 | [-3.36, 7.90] |
| mixed_corrupted | mean_avg_speed_mps | max_pressure | -9.83 | [-14.89, -4.92] |
| mixed_corrupted | mean_avg_speed_mps | schema_v3 | -1.75 | [-6.77, 2.38] |
| mixed_corrupted | mean_global_queue | native_sumo | 6.89 | [-4.06, 17.58] |
| mixed_corrupted | mean_global_queue | max_pressure | -15.15 | [-27.91, -4.95] |
| mixed_corrupted | mean_global_queue | schema_v3 | -2.94 | [-17.50, 7.33] |
| mixed_corrupted | mean_global_wait | native_sumo | -41.88 | [-85.99, -6.52] |
| mixed_corrupted | mean_global_wait | max_pressure | -42.80 | [-83.03, -13.39] |
| mixed_corrupted | mean_global_wait | schema_v3 | -12.51 | [-55.02, 14.71] |

Each campaign's statistical_summary.csv contains all map/rate/metric comparisons, paired confidence intervals, seed wins/losses, and Holm-adjusted condition p-values.
Use the per-condition rows when assessing variation; the repository's pooled rows treat repeated map/rate observations as separate pairs and should not be used as independent-city evidence.

Limits: ordinary traffic only; the historical v3 and MaxPressure baselines have more state information. V3 was trained on a different corpus, so this is not an isolated sensing ablation.
The archived v4 spillback/starvation collector reads the wrong adapter key; its zero fields are unavailable measurements. Collision counts, ambulance trip times and per-vehicle trip delay are not measured by this harness.
A data audit passing does not mean the learned controller outperforms a baseline.
