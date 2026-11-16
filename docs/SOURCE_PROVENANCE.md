# Source provenance and reproducibility boundary

The Python source code, tests, and configurations in this repository are byte-for-byte copies of the historical
`code/SymbioNet/` snapshot inside `SymbioNet_交接_核心_1115c.zip`, which the project's handover record identifies
as the original training source on the Legion workstation at `G:\SymbioNet`.

No algorithmic code, training setting, random seed, or experimental procedure was edited for publication.
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

The repository additionally provides a Windows/Linux quick start, a non-training validation script, a
SHA-256 manifest of the archived original files, and lightweight GitHub Actions checks. These additions
were not part of the original experiment, and do not change the original model, training settings,
random seeds, or numerical results. GitHub documentation may be improved independently of the original
training code. If a source-file change is needed later, its impact on the paper and historical code hashes
must be reviewed and disclosed rather than silently resetting the manifest.
