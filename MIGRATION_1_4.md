# SPINE 1.4 configuration migration

Maintained configurations require SPINE **1.4.0 or later**. The default in
`DEFAULT_SPINE_VERSION` is `1.4.0`; both interactive and scheduler launchers check
`--version` on the selected executable after environment setup, inside the
container when selected. This also checks explicit source-checkout executables.
The guard fails before processing inputs when the runtime is too old or cannot
report its version.

The migration follows the tagged [configuration reference](https://github.com/DeepLearnPhysics/spine/blob/v1.4.0/src/spine/config/README.md)
and [release notes](https://github.com/DeepLearnPhysics/spine/releases/tag/v1.4.0).
Publication was verified on October 6, 2026: PyPI provided the
[1.4.0 wheel and source distribution](https://pypi.org/project/spine/1.4.0/),
and GHCR returned the `ghcr.io/deeplearnphysics/spine:1.4.0` manifest with digest
`sha256:bd80f0d5723c43fa111f088cf1237c1e52952ecc90a078d7b228ea84fc689d3b`.
The container manifest was checked; a GPU container workload was not run locally.

## Descriptors and ordered stages

Implementation selectors use `provider`. Metadata names, instance identities,
detector fields, and the loader's `collate_fn` field keep their meanings. Parser
schemas remain mappings keyed by output product. A descriptor uses inline
parameters or a nested `config`, never both.

Full-chain, post-processing, analysis, augmentation, SBND model calibration, and
ProtoDUNE-SP model calibration use ordered `stages`. Names are unique within each
list; omit `provider` when it equals `name`. The previous resolved priority order
(including insertion order for ties) is now explicit. Stages have no `priority`.
Manager settings such as analysis `overwrite` remain outside the list.

```yaml
post:
  stages:
    - name: direction
      run_mode: both
    - name: calorimetry
      provider: calo_ke
      config:
        scaling: 1.0
```

Apply a modifier after its base configuration. Match the target's inline or
nested parameter form:

```yaml
override:
  post.stages~:
    - update:
        name: calorimetry
        changes:
          config:
            scaling: 1.1
    - insert:
        after: calorimetry
        value:
          name: fiducial
          margin: 25.0
          mode: meta
    - remove:
        name: direction
```

An edit requires its list, target, and insertion anchor to exist. It cannot use
`optional_paths`. A later replacement of an entire list discards earlier edits.
Named updates merge parameters; they cannot delete individual fields. Where a
parameter must disappear, the modifier removes and reinserts the complete stage
at its original position. Setting a parameter to `null` does not delete it.

Overrides run before removals at every include boundary. The prepared UResNet
minibatch and cached-training input cleanup is included: redundant deletions are
removed while intended overrides and unrelated deletions remain. Validation has
not been relaxed.

## Calibration exceptions that preserve composition

Two calibration cases deliberately retain the supported mapping form:

- **ICARUS in-model calibration.** Systematic modifiers compose priority-10
  response/gain/field corrections with priority-9 smearing. Their order must stay
  correct in either modifier order. Named edits cannot conditionally anchor a
  correction before smearing when it exists and before gain otherwise. Retaining
  these mappings and their priorities preserves all tested combinations. Their
  implementation selectors still use `provider`.
- **Calibration inside a post-processing stage.** The outer post manager uses
  `stages`, but its calibration corrections remain a mapping. SPINE 1.4 cannot
  traverse an outer named list to edit an inner named list. Replacing the entire
  inner list for every lifetime/transparency/data modifier would overwrite other
  composed corrections. Outer `post.stages~` updates merge these ordinary nested
  calibration parameters safely.

These are explicit remaining schema migrations, pending richer SPINE named-edit
support or a separately designed composition scheme. Do not convert them by
mechanically replacing mapping entries with lists.

The existing optional SBND `predeghost_scale` modifier still defaults to **1.03**;
no production base enables it. Its embedded gain correction uses `stages`.
See the [SBND example](config/infer/sbnd/README.md#pre-deghosting-charge-scale)
for overriding the factor. The migration leaves enabled calibrations and their
parameters unchanged. It relies on SPINE's preserved charge/hit handling between
deghosting and subsequent calibration; the model implementation is not changed
in this repository.

## Data modifier compatibility

The two prepared `optional_paths: [post.time_containment.run_mode]` declarations
are superseded by scoped named-list edits. Common fragments handle IO/model
changes; dated modifiers carry the appropriate post stage operations.

| Pipeline post version | Data modifier |
| --- | --- |
| ICARUS `250303` | `data:250303` |
| ICARUS `>=250625` | `data:250625` |
| SBND `<=250328` (model `<260501`) | Existing `data:240720` or `data:250328`, subject to their other constraints |
| SBND `>=250818` (model `<260501`) | New `data:250818`, or `data:250901` for its existing calibration choice |
| SBND model `>=260501` | Existing `data:260501` |

The older ICARUS data modifiers retain their previous version constraints.
Twenty-one previously accepted cross-version combinations are now rejected
because their stage sets or deleted fields differ. The regression fixture lists
every affected base/modifier pair. Prefer automatic compatible modifier selection
or pin the version appropriate to the table. No named-list target is optional.

Six archived wrappers include maintained fragments: five ICARUS `250625` data
wrappers now reference `data:250625`, and the SBND `250818` data wrapper references
`data:250818`. Their resolved production settings are preserved. Other archived
configuration files are unchanged.

The analyzer fragment `test/common/full_chain/analyzers_v1.yaml` must be loaded
with a full-chain base containing `post.match`; a standalone fragment is not an
executable analysis configuration.

## Validation and reproducibility

The pre-conversion snapshot included all 13 prepared edits. The initial inventory
was 685 tracked YAML files, including 78 archived wrappers. The maintained sweep
examined 607 YAML files and all dated inference/modifier pairs: 371 combinations
resolved before conversion. Of those, 350 retain identical normalized behavior;
the 21 deliberate compatibility rejections are described above. Five additional
combinations use the new SBND data modifier. The audit also covers 176 supported
two-modifier compositions, in both orders, on the latest ICARUS/SBND bases.

`tests/fixtures/spine140_baseline.json` stores pre-conversion fingerprints, with
normalization in `tests/config_migration.py`. The comparison retains constructor
values, provider identities, execution order, model and loss inputs, and output
settings. It normalizes schema spelling and relocatable paths, leaving download
tags unresolved to avoid fetching weights or calibration databases. The 1,035
retained regression cases include the 21 expected strict rejections and all 78
archived wrappers, whose normalized settings are unchanged. Nineteen
incomplete standalone fragments that previously warned are excluded from this
fixture; their intended compositions are tested. The bare analyzer fragment is
also excluded. Resolved regression compositions treat warnings as errors.

The tests additionally construct real calibration, post-processing, analysis,
and augmentation managers from the published 1.4.0 package, execute and invert
the SBND gain, run subsequent SBND calibration, and execute ND-LAr augmentation
on synthetic sparse data. Existing inference/training/cache and archived-wrapper
tests remain part of the full suite. Run in an environment with SPINE installed:

```bash
python -m pip install -r requirements-dev.txt 'spine==1.4.0'
python -m pytest --no-cov
pre-commit run --all-files
```

Local validation completed with **4,216 passed and 6 skipped**; the skips are
existing inventory/composite-test cases, and all new manager smoke tests ran.
All repository pre-commit hooks passed on the changed files.

Manager tests need the optional CPU runtime dependencies, including PyTorch;
they skip when PyTorch is unavailable. Full neural-network inference/training,
ROOT/LArCV ingestion, external calibration databases, optical matching, and GPU
execution require production dependencies and data and were not run locally.
Configuration equivalence and manager smoke tests do not establish end-to-end
numerical reconstruction equivalence on detector events.
