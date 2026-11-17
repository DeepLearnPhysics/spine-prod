# DUNE-VD 10 kt module 1x8x6 training candidate

`full_chain_261117.yaml` derives from the latest DUNE-HD training candidate,
`dune-hd-10kt-1x2x6/full_chain_261022.yaml`. This is a candidate for review and a
production smoke test, not a released VD model. No pretrained VD weights are
published or selected automatically.

## What the pipeline does

The pipeline expands to 21 stages: scan/build train and validation filters, train
six networks, materialize four successive cache products for each split, assemble
the best component weights, evaluate held-out events, and produce a metric report.

VD bundles include detector-local snapshots of the required HD recipe components,
plus existing `common` fragments. The snapshot filenames retain their source dates
for provenance; they are dependencies of this candidate, not historical VD releases.
Architectures, losses, parsing, and training policy remain auditable against HD.
VD geometry and calibration are overridden explicitly. Future VD-specific changes
belong in these VD bundles rather than edits to the HD recipes.

All model-bearing stages use the geometry already selected by VD truth conversion:

```yaml
geo:
  detector: dune-vd-10kt-1x8x6
  tag: dunevd10kt_3view_30deg_v3_refactored_1x8x6ref
```

Raw and cached loaders preserve whole logical events (`collate_fn: all`) so cached
raw indices and graph products remain aligned. The first half of the validation
list's event domain is used for training validation; the second half is reserved
for final evaluation. Caches retain the complete domain.

The candidate retains HD's six PID classes, reconstructed-object overlap threshold
of 0.5 for PID/primary supervision, closest-fragment primary policy, and explicit
orientation targets. PPN has a 100-epoch ceiling; the other training recipes retain
their HD ceilings and early-stopping policy. No augmentation is newly enabled.

Changes to scheduling relative to HD:

- Scan stages explicitly request two hours, following the ICARUS scan-time lesson.
- Evaluation groups inputs across four tasks instead of creating one task per file.
- Other stage limits and batches remain HD starting values pending VD measurements.

## Inputs and launch

The candidate defaults to the supplied VD MPV/MPR v0 dataset:

- `/sdf/data/neutrino/dune/sim/vd/mpvmpr_v0/train_file_list.txt`
- `/sdf/data/neutrino/dune/sim/vd/mpvmpr_v0/test_file_list.txt`

Use `--source-list` and `--val-source-list` to override them consistently across all
stages. These are VD inputs; no HD dataset is selected automatically.

From the repository root, after reviewing calibration and input compatibility:

```bash
python3 submit.py \
  --pipeline pipelines/dune-vd-10kt-1x8x6/full_chain_261117.yaml \
  --workspace /path/to/runs/dune-vd-261117 \
  --source-list /sdf/data/neutrino/dune/sim/vd/mpvmpr_v0/train_file_list.txt \
  --val-source-list /sdf/data/neutrino/dune/sim/vd/mpvmpr_v0/test_file_list.txt \
  --dry-run
```

Remove `--dry-run` to submit. A workspace name does not shorten the training budget.
Use a fresh workspace/cache repository for VD; HD caches contain geometry- and
calibration-dependent products and must not be reused. An optional HD warm start
requires an explicit `--warm-start` checkpoint and should be compared with training
from scratch; compatible tensor shapes do not establish detector transfer quality.

## Detector differences requiring review

### 1. Calibration: provisional, not established for VD

`config/model/dune-vd-10kt-1x8x6/calibration/calibration_261117.yaml` contains the
explicit HD baseline, pending confirmation against the VD simulation/reconstruction
FHiCL used to produce the selected input files:

| Setting | Inherited HD value | VD question |
| --- | --- | --- |
| Gain | 198.21605550049554 electrons/ADC | Is the collection-charge conversion the same? |
| Recombination | Modified box | Does the sample use the same model/parameters? |
| Electric field | 0.5 kV/cm | Is this the nominal field used in this simulation? |
| MIP dE/dx | 2.105168 MeV/cm | Is this charge-to-energy assumption appropriate? |
| Lifetime | 10400 microseconds | What lifetime/attenuation was applied to this sample? |
| Drift velocity | 0.160563 cm/microsecond | Does it match VD field/temperature and production settings? |

Gain is applied before recombination and lifetime correction. The same VD file is
used by the deghosting cache producer, exported model, and evaluator. Postprocessing
does not repeat calibration. Update this one file when the VD constants are known;
rebuild affected caches after any calibration change.

### 2. Geometry, coordinates, and source IDs

Confirm the sample is the `1x8x6` reference layout and uses the same coordinate
frame and source-ID convention as the selected geometry tag. SPINE 1.4.5's geometry
file specifies the up direction and drift direction along **x** for this reference
layout. Do not rotate x/y solely because the detector is called vertical drift.

The geometry maps the 48 detector/TPC IDs into one reference volume. Verify the
`sparse3d_reco_cryo` and `sparse3d_reco_tpc` values against that mapping, particularly
for lifetime corrections and points near anode/cathode boundaries. Also check voxel
size, bounding boxes, offsets, and units; the model's voxel-scale distances otherwise
inherit HD assumptions.

### 3. Reconstruction products and collection plane

Truth conversion documents shared HD/VD truth products, but that does not establish
reconstruction-product equivalence. The inherited raw schema expects:

- `sparse3d_reco` and `sparse3d_reco_chi2` for deghosting.
- `sparse3d_reco_hit_charge0/1/2` and `sparse3d_reco_hit_key0/1/2` for charge rescaling.
- `sparse3d_reco_cryo` and `sparse3d_reco_tpc` for source IDs.
- `sparse3d_pcluster`, ghost/semantic labels, cluster products,
  `particle_corrected`, and `neutrino_mpv` for supervision and metrics.

Check that the collection-plane identification, wire-angle conventions, charge units,
and shared-hit keys are correct for VD. The inherited `charge_rescaling: collection`
policy is not a validation of the VD plane layout. Wire geometry and ghost-label
quality can change deghosting difficulty substantially.

### 4. Dataset composition and truth policy

The supplied dataset is VD MPV/MPR v0. Confirm its MPV/MPR mixture, particle species (including the sixth PID
class), energy ranges, multiplicities, ghost-label quality, and train/test separation.
Inherited truth exclusions for secondary/MPR PID and primary labels need the same
meaning in the VD production. Check that orientation labels are usable and that the
validation/evaluation halves have sufficient statistics and no event duplication.

### 5. Occupancy, event rejection, and resources

The inherited filter rejects events with **500,000 or more** reconstructed space
points. Measure rejection rate versus multiplicity/energy before accepting this
threshold: geometry and wire ambiguities can alter VD occupancy and selection bias.

| Training stage | HD starting batch convention | Walltime | Profile |
| --- | --- | --- | --- |
| Deghosting | Total batch 12 | 5 days | S3DF Ampere full node |
| UResNet-PPN | Minibatch 32 | 3 days | S3DF Ampere full node |
| Graph-SPICE | Minibatch 32 | 2 days | S3DF Ampere full node |
| GrapPA shower | Minibatch 16 | 24 hours | S3DF Ampere |
| GrapPA track | Minibatch 32 | 16 hours | S3DF Ampere |
| GrapPA interaction | Minibatch 128 | 8 hours | S3DF Ampere |

Cache jobs retain eight-hour limits, four training tasks and one validation task;
evaluation uses four tasks with eight-hour limits. Measure GPU peaks, host RAM,
cache size/I/O, scan throughput, and step latency before tuning. The same point cap
does not guarantee the same graph size or memory cost as HD. No VD resource timings
are claimed yet.

### 6. Postprocessing and acceptance

The HD postprocessing remains the starting point, including module fiducial margins
`[[25,25], [25,25], [30,50]]` cm, calorimetry, range/MCS estimators, topology thresholds,
and vertex/orientation policy. Review these margins in the actual VD coordinate frame
and check containment/acceptance near its longer drift volume and boundaries.

This is charge reconstruction and metric evaluation; it does not add a VD optical
matching calibration or an optical reconstruction validation campaign.

## Validation before PR/production

Configuration tests cover all stages resolving, the installed VD geometry, HD
architecture/objective inheritance, cached-target agreement, raw-index-safe collation,
source overrides, held-out evaluation, and weight export/report wiring. A calibration
mutation test verifies that a VD-only correction reaches cache/export/evaluation.

These checks cannot validate sample-specific calibration, ROOT products, geometry
alignment, GPU performance, or physics quality. Before production, inspect a small
VD sample and run scan/filter, a short deghosting/cache pass, and downstream cached
training/evaluation checks. Then review resource measurements and metrics.

The work is isolated on `feature/dune-vd-staged-training`, based on `origin/main`.
The ICARUS migration remains on `feature/icarus-model-pipeline-migration`; switching
local branches does not control the already-running remote ICARUS test. Open a PR
only after the detector assumptions above are reviewed.
