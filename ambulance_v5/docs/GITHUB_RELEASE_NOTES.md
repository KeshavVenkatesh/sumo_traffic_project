# Ambulance v5 round-40 research artifacts

This prerelease preserves the reviewed Ambulance v5 round-40 model, training
evidence, map/demand corpus, frozen benchmark inputs, and complete 1,080-run
Native SUMO versus MaxPressure versus learned-controller evaluation.

## Status

- Checkpoint label: `unpromoted_final_round_40`
- No Ambulance v5 `_best.pt` exists
- Validation-promotion gate: not passed
- Strict safety gate: not passed
- Final 1,080-run test matrix: frozen

## Main result

Mean ambulance response time improved in all 12 unseen map/rate conditions.
The pooled reduction was approximately 21.52% versus Native SUMO and 14.31%
versus MaxPressure.

Pooled ordinary-traffic differences versus MaxPressure were not statistically
significant, although San Diego at rate 6 was a negative ordinary-traffic
condition.

Use `RELEASE_ASSET_SHA256SUMS.txt` to verify all downloaded archives. See
`AMBULANCE_V5_REPOSITORY_GUIDE.md` for the complete file map and interpretation
caveats.
