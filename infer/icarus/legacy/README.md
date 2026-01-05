# Legacy Pre-Composed Configurations

⚠️ **DEPRECATED**: These configurations are maintained for reproducibility only.

## Migration Guide

For new work, use base configurations with modifiers instead of these pre-composed variants.

### How to Migrate

Instead of using these legacy configs, compose them dynamically:

**Old way (legacy):**
```bash
spine-prod --config icarus/legacy/icarus_full_chain_data_co_lite_250625.yaml --files data/*.root
```

**New way (recommended):**
```bash
spine-prod --config icarus/icarus_full_chain_co_250625 --modifier data,lite --files data/*.root
```

### Migration Examples

| Legacy Config | Base Config | Modifiers |
|--------------|-------------|-----------|
| `icarus_full_chain_data_co_lite_250625.yaml` | `icarus_full_chain_co_250625` | `data,lite` |
| `icarus_full_chain_single_co_250625.yaml` | `icarus_full_chain_co_250625` | `single` |
| `icarus_full_chain_co_numi_250625.yaml` | `icarus_full_chain_co_250625` | `numi` |
| `icarus_full_chain_data_co_unblind_lite_250625.yaml` | `icarus_full_chain_co_250625` | `data,lite,unblind` |
| `icarus_full_chain_co_4ms_lite_250625.yaml` | `icarus_full_chain_co_250625` | `4ms,lite` |
| `icarus_full_chain_co_8ms_lite_250625.yaml` | `icarus_full_chain_co_250625` | `8ms,lite` |

## Available Modifiers

- **`data`**: Transform to data-only mode (removes truth labels)
- **`lite`**: Enable lite-output mode (removes heavy data products)
- **`single`**: Process single-cryostat simulations
- **`numi`**: Use NuMI-specific settings (wider flash matching window)
- **`unblind`**: Process only unblinded data events
- **`4ms`**: Use 4ms electron lifetime calibration
- **`8ms`**: Use 8ms electron lifetime calibration
- **`transp`**: Apply transparency corrections
- **`lowbs`**: Use low batch size for memory-constrained environments

See `../modifier/` directory for available modifiers and their implementations.

## Why Deprecated?

Pre-composed configurations lead to:
- **Combinatorial explosion**: N models × 2^M modifiers = too many files
- **Maintenance burden**: Changes to modifiers require updating multiple files
- **Unclear composition**: Hard to see what modifiers are applied from filename
- **Duplication**: Same modifier logic repeated across multiple configs

The new approach with dynamic composition solves all these issues.

## Deprecation Policy

These configs will be:
1. **Maintained** for existing production runs and reproducibility
2. **Not updated** with new features or bug fixes
3. **Removed** after 12 months (around January 2027) or when no longer in use

For questions, see the main ICARUS README: `../README.md`
