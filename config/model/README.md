# SPINE Model Configurations

This directory owns model definitions shared by training and inference. Model
files must not own datasets, training schedules, deployed checkpoint paths, or
post-processing settings.

The tree separates two independent forms of versioning:

- `common/` contains detector-independent model structures tied to a SPINE
  model-schema generation, such as `base_v1.yaml` and `network_v1.yaml`.
- detector directories such as `generic/` contain immutable dated model
  revisions which derive from a pinned common structure and supply all
  detector-dependent parameters.

For example:

```yaml
include: model/generic/graph_spice/model_240805.yaml
```

Imports are rooted at `config/` through `SPINE_CONFIG_PATH`. Source
`configure.sh` before loading or submitting configurations so that SPINE can
resolve these paths.

Common structures are not mutable aliases. An incompatible SPINE model-schema
change creates a new common version rather than changing `base_v1.yaml` in
place. Detector revisions use the `YYMMDD` convention and derive directly from
their common version instead of chaining through previous detector revisions.

Component versions evolve independently. For generic simulations, UResNet
remains at revision `240718`, while UResNet-PPN and Graph-SPICE have `240805`
revisions because their configurations changed. A future full-chain model will
pin the required revision of each component explicitly.

UResNet-PPN composes the common UResNet network and loss because UResNet is its
segmentation component. Graph-SPICE instead owns a dedicated UResNet embedder
configuration, following SPINE's canonical Graph-SPICE example; its input,
kernel, and required spatial extent differ from standalone UResNet.
