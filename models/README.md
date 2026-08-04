# Checkpoints

Model binaries are distributed through the
`schema-v3-checkpoints-20260727` GitHub Release rather than Git LFS.

From the repository root, download them with:

    gh release download schema-v3-checkpoints-20260727 \
      --repo KeshavVenkatesh/sumo_traffic_project \
      --dir models

Verify the downloaded files with:

    (cd models && sha256sum -c SHA256SUMS)

The deployment checkpoint used by the completed July 30, 2026 evaluation is:

    map_agnostic_multiagent_v3_best.zip

Its SHA-256 digest is:

    be832797883d589441fd27d52b87884320a624af024eab0dac4ed8d9860f0706

Use the sibling `*_map_agnostic.json` file as its runtime schema metadata.
The `*_trainer.pt` file is optimizer/resume state and is not required for
inference.
