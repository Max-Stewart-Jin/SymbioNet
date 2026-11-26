# Source provenance and reproducibility boundary

The initial public repository preserved the historical `code/SymbioNet/` snapshot from
`SymbioNet_交接_核心_1115c.zip`, identified by the handover record as the experiment source on the Legion workstation.
The public release preserves `symbionet/` (model, data, engine), `formal.py`, `train.py`, and the experiment training settings.
Optional historical screening OA references were removed from `configs/formal.yaml`; the corresponding unit test
uses synthetic reference scores. These changes do not affect the training or evaluation algorithm.
For a cleaner repository layout, six auxiliary launchers/helpers were moved to `scripts/`, with only
project-root resolution, imports, and internal source-path references adjusted. No changes were made to their
model equations, loss functions, experiment parameters, split indices, or random seeds. Existing auxiliary
commands now use `python -m scripts.<name>` from the repository root.

`docs/ORIGINAL_SNAPSHOT_SHA256.json` records the **historical** source file paths and hashes, while
`docs/RELEASE_SOURCE_SHA256.json` lists the relocated files and verifies the **current** source checkout.
The relocation and removal of screening result values are documented; the original snapshot manifest remains unchanged.
Five workstation-only diagnostic/orchestration scripts (`gate_check.py`, `go_rest.py`, `judge.py`, `robref.py`,
`seed_merge.py`) are intentionally omitted because they are not needed to execute the published protocol.
Repository documents (`README.md`, `.gitignore`, `docs/`, `CITATION.cff`, `splits/README.md`) were prepared separately
and are not claimed to be byte-identical to the historical training snapshot.

This comparison is to the saved handover snapshot, not to a live copy of the author's computer; the live machine
is not available to this publication environment. Archived output JSON files contain source-file hash prefixes
and are retained by the authors for provenance checking. The current source release has not been subject to a
fresh full GPU re-run. Previous original training outputs remain the basis of the manuscript.

The original experiment sensor data and third-party model source repositories are not redistributed. The provided
fixed Geo-ΔSSM split indices were approved for sharing by a collaborator, as conveyed by the project author.

## Publication-only improvements

The repository also provides a Windows/Linux quick start, non-training validation scripts, historical and
current source SHA-256 manifests, and lightweight GitHub Actions checks. These additions
were not part of the original experiment, and do not change the original model, training settings,
random seeds, or numerical results. GitHub documentation may be improved independently of the original
training code. If a source-file change is needed later, its impact on the paper and historical code hashes
must be reviewed and disclosed rather than silently resetting the manifest.
