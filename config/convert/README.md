# SPINE Production Conversion Configurations

This directory contains standalone bundles that convert source data into useful
SPINE products without running a trained reconstruction model.

The current `truth_YYMMDD.yaml` bundles convert LArCV simulation files into
SPINE HDF5 files containing built truth objects. These outputs can be inspected
directly with tools such as spinal-tap.

Use an explicitly dated bundle for reproducible production:

```bash
./submit.py --config convert/protodune-sp/truth_260210.yaml \
  --source input.root
```

The detector directory and its `/latest` alias both select the newest complete
conversion bundle for convenience:

```bash
./submit.py --config convert/protodune-sp --source input.root
./submit.py --config convert/protodune-sp/latest --source input.root
```

`latest` selects a complete dated bundle. It does not independently combine the
newest detector IO and base fragments, which could produce an unreviewed pairing.

A detector has another dated truth bundle only when a base or IO revision changes
the converted artifact—for example its geometry, truth parsing, built objects, or
written products. Reco-only and otherwise irrelevant configuration changes reuse
the existing conversion bundle.
