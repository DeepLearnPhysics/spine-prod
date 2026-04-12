# Common Configuration Directory

This directory contains **universal, detector-agnostic, mode-agnostic** configuration files shared across all SPINE workflows. These configurations define core model architectures and other reusable components that remain consistent regardless of detector type or execution mode (training/inference).

## Purpose

The `config/common/` hierarchy enables:

1. **Single Source of Truth**: Model architectures defined once and reused everywhere
2. **Consistency Guarantee**: Ensures training and inference use identical model definitions
3. **Multi-Detector Efficiency**: New detectors inherit proven architectures automatically
4. **Reduced Duplication**: Avoid maintaining separate copies of the same model configs
5. **Clear Divergence Tracking**: Easy to identify when/where detectors or modes deviate from defaults

## Directory Structure

```
config/common/
├── model/                      # Universal model architectures
│   ├── graph_spice/           # Graph-SPICE clustering model
│   │   ├── core_YYMMDD.yaml
│   │   └── loss_YYMMDD.yaml
│   ├── grappa_inter/          # GrapPA interaction aggregation
│   ├── grappa_shower/         # GrapPA shower aggregation
│   ├── grappa_track/          # GrapPA track aggregation
│   ├── uresnet_deghost/       # UResNet deghosting
│   ├── uresnet_ppn/           # UResNet + PPN (point proposal)
│   └── full_chain/            # Complete full-chain architecture
│       └── architecture_YYMMDD.yaml
└── post/                      # Universal post-processing (if applicable)
    └── post_common_YYMMDD.yaml
```

## What Belongs Here

**DO place in `config/common/`:**
- Model network architectures (layers, filters, depths, reps)
- Core hyperparameters that define network structure
- Activation and normalization layer definitions
- Module connectivity and chaining logic
- Loss function definitions (when universal)
- Post-processing algorithms that work across all detectors

**DO NOT place here:**
- Detector-specific calibration parameters
- Weight/checkpoint paths
- Training hyperparameters (learning rate, batch size, epochs)
- Dataset paths or I/O configurations
- Detector geometry or timing parameters
- Inference-specific runtime settings

## Configuration Resolution Hierarchy

SPINE resolves configurations with the following priority (most specific to most general):

```
1. config/{mode}/{detector}/{component}/     ← Detector + mode specific
2. config/{mode}/common/{component}/         ← Mode-specific, detector-agnostic
3. config/common/{component}/                ← Universal (this directory)
```

**Example resolution for ICARUS inference:**
```yaml
# config/infer/icarus/model/model_260202.yaml
include:
  - ../../../common/model/full_chain/architecture_260202.yaml  # Universal architecture

override:
  model.weight_path: https://s3df.slac.stanford.edu/.../icarus_weights.ckpt
  model.modules.calibration.gain: 250.0  # ICARUS-specific
```

## Versioning Scheme

All configuration files follow the `YYMMDD` versioning pattern:
- `architecture_260202.yaml` → February 2, 2026 version
- Versions are date-based for chronological tracking
- Use `latest` when referring to the most recent config in production scripts

## Usage Examples

### Training Configuration (References Common)

```yaml
# config/train/dune10kt-1x2x6/full_chain/full_chain_graph_spice.yaml
include:
  - ../../../../common/model/graph_spice/core_260202.yaml      # Universal
  - ../../../../common/model/graph_spice/loss_260202.yaml      # Universal
  - ../base/base_260202.yaml                                   # Detector-specific
  - ../io/io_260202.yaml                                       # Detector-specific

override:
  base.log_dir: /sdf/data/.../dune/spine/train/...
  io.loader.batch_size: 48
```

### Inference Configuration (References Common)

```yaml
# config/infer/sbnd/model/model_260202.yaml
include: 
  - common/model/full_chain/architecture_260202.yaml  # SPINE_CONFIG_PATH-relative

override:
  model.weight_path: !download
    url: https://s3df.slac.stanford.edu/.../sbnd_weights.ckpt
    hash: abc123...
  model.modules.calibration.gain: 198.0  # SBND-specific
```

## Best Practices

### When Creating New Architectures

1. **Start Universal**: Define the model in `common/model/` first
2. **Test Broadly**: Validate architecture works across multiple detectors
3. **Document Changes**: Update `__meta__` blocks with clear descriptions
4. **Version Appropriately**: Create new dated files for significant changes

### When Modifying Existing Architectures

1. **Consider Impact**: Changes here affect ALL detectors and modes
2. **Preserve Compatibility**: Maintain backward compatibility when possible
3. **Communicate**: Document breaking changes in CHANGELOG
4. **Version Bump**: Create new versioned file rather than editing in-place

### When Detector-Specific Override Needed

1. **Override, Don't Duplicate**: Reference common config, override specific keys
2. **Document Rationale**: Explain why detector diverges from common
3. **Consider Refactoring**: If many detectors override the same thing, move it to common

## Migration Guidelines

When moving configs to this directory:

1. **Identify Common Elements**: Extract truly universal architecture definitions
2. **Strip Specifics**: Remove detector, dataset, and weight-specific parameters
3. **Update References**: Update detector configs to include from common
4. **Test Thoroughly**: Verify all affected workflows still function correctly
5. **Document Changes**: Update individual detector READMEs

## Related Documentation

- [Inference Configurations](../infer/README.md)
- [Training Configurations](../train/README.md)
- [SPINE Configuration System](../../spine/docs/configuration.md)
- [Version Management Guide](../../VERSION_MANAGEMENT.md)

## Maintenance

**Maintainers**: SPINE Development Team  
**Review Required**: Changes to common configs require multi-detector validation  
**Update Frequency**: As needed for architectural improvements, typically with major releases

---

For questions or issues with common configurations, please open an issue in the [spine-prod repository](https://github.com/DeepLearnPhysics/spine-prod).
