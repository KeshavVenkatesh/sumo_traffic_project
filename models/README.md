# Checkpoints

Model binaries are distributed through the
`schema-v3-checkpoints-20260727` GitHub Release rather than Git LFS.

From the repository root, download them with:

    gh release download schema-v3-checkpoints-20260727 \
      --repo KeshavVenkatesh/sumo_traffic_project \
      --dir models

Verify the downloaded files with:

    (cd models && sha256sum -c SHA256SUMS)
