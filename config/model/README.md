# SPINE Model Configurations

This directory owns model definitions shared by training and inference. Model
files must not own datasets, training schedules, deployed checkpoint paths, or
post-processing settings.

Each model directory may contain:

- reusable network and loss fragments inserted with `!include`;
- a `default.yaml` fragment defining the standalone model interface; and
- detector-specific variants only when the architecture genuinely differs.

Training and inference bundles include these definitions independently. They
do not inherit from one another.

Imports are rooted at `config/` through `SPINE_CONFIG_PATH`, for example:

```yaml
include: model/uresnet/default.yaml
```

Source `configure.sh` before loading or submitting configurations so that the
repository configuration root is available to SPINE.

The generic full-chain configuration uses the default UResNet architecture, so
there is intentionally no `model/generic/uresnet.yaml` override. A detector
variant should be introduced only when it changes a model-owned parameter.
