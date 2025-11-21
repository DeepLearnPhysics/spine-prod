# Summary of ICARUS full chain configurations and their characteristics

The configurations below have been trained on ICARUS MPV/MPR datasets. This summary is divided by training/validation dataset.

## Configurations for MPV/MPR v02

These weights have been trained/validated using the following files:
- Training set: `/sdf/data/neutrino/icarus/sim/mpvmpr_v2/train_file_list.txt`
- Test set: `/sdf/data/neutrino/icarus/sim/mpvmpr_v2/test_file_list.txt`

### July 19th 2024

```shell
icarus_full_chain_240719.cfg
icarus_full_chain_single_240719.cfg
icarus_full_chain_numi_240719.cfg
icarus_full_chain_data_240719.cfg
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Class-weighted loss on PID predictions
  - The `*_single_*` declination works on single-cryostat simulations
  - The `*_numi_*` declination has a wider flash matching window (10 us)
  - The `*_data_*` declination is tailored for data (no labels)

Known issue(s):
  - The shower start point prediction of electron showers is problematic due to the way PPN labeling is trained

### August 12th 2024

```shell
icarus_full_chain_240812.cfg
icarus_full_chain_single_240812.cfg
icarus_full_chain_numi_240812.cfg
icarus_full_chain_data_240812.cfg
icarus_full_chain_data_numi_240812.cfg
icarus_full_chain_co_240812.cfg
icarus_full_chain_data_co_240812.cfg
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Class-weighted loss on PID predictions
  - The `*_single_*` declination works on single-cryostat simulations
  - The `*_numi_*` declination has a wider flash matching window (10 us)
  - The `*_data_*` declination is tailored for BNB data (no labels)
  - The `*_data_numi_*` declination is tailored for NuMI data (no labels)
  - The `*_co_*` declination only uses collection charge
  - The `*_data_co_*` is tailored for data and only uses collection charge

Known issue(s):
  - Resolves the issue with the PPN target in the previous set of weights
  - Removed PPN-based end point predictions
  - The signal gain on the first induction plane is wrong (`_fitFR` fcl file)
  - No other known issue


## Configurations for MPV/MPR v03

These weights have been trained/validated using the following files:
- Training set: `/sdf/data/neutrino/icarus/sim/mpvmpr_v3/train_file_list.txt`
- Test set: `/sdf/data/neutrino/icarus/sim/mpvmpr_v3/test_file_list.txt`

### January 15th 2025 

```shell
icarus_full_chain_250115.cfg
icarus_full_chain_co_250115.cfg
icarus_full_chain_co_lite_250115.cfg
icarus_full_chain_numi_250115.cfg
icarus_full_chain_co_numi_250115.cfg
icarus_full_chain_single_250115.cfg
icarus_full_chain_single_co_250115.cfg
icarus_full_chain_co_4ms_lite_250115.cfg
icarus_full_chain_co_8ms_lite_250115.cfg
icarus_full_chain_data_250115.cfg
icarus_full_chain_data_co_250115.cfg
icarus_full_chain_data_co_lite_250115.cfg
icarus_full_chain_data_co_unblind_lite_250115.cfg
icarus_full_chain_data_numi_250115.cfg
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Class-weighted loss on PID predictions
  - The `*_co_*` declinations only use collection charge
  - The `*_numi_*` declinations are tailored for NuMI (10 us beam window for FM)
  - The `*_data_*` declinations are tailored for data (no labels)
  - The `*_lite_*` declinations output directly to lite files
  - The `*_single_*` declination works on single-cryostat simulations
  - The `*_unblind_*` declination only processes unblinded data

Known issue(s):
  - Resolves the issue with the first induction plane gain
  - Uses correct calibration constant (courtesy of Lane Kashur)
  - No other known issue

### March 3rd 2025 

```shell
icarus_full_chain_co_250303.cfg
icarus_full_chain_co_lite_250303.cfg
icarus_full_chain_co_numi_250303.cfg
icarus_full_chain_single_co_250303.cfg
icarus_full_chain_co_4ms_lite_250303.cfg
icarus_full_chain_co_8ms_lite_250303.cfg
icarus_full_chain_co_transp_lite_250303.cfg
icarus_full_chain_data_250303.cfg
icarus_full_chain_data_co_250303.cfg
icarus_full_chain_data_co_lite_250303.cfg
icarus_full_chain_data_co_unblind_lite_250303.cfg
icarus_full_chain_data_co_numi_250303.cfg
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Class-weighted loss on PID predictions
  - The `*_co_*` declinations only use collection charge
  - The `*_numi_*` declinations are tailored for NuMI (10 us beam window for FM)
  - The `*_4ms_*` declinations are tailored for 4ms lifetime variations
  - The `*_8ms_*` declinations are tailored for 8ms lifetime variations
  - The `*_transp_*` declinations are tailored for the YZ transparency variations
  - The `*_data_*` declinations are tailored for data (no labels)
  - The `*_lite_*` declinations output directly to lite files
  - The `*_single_*` declination works on single-cryostat simulations
  - The `*_unblind_*` declination only processes unblinded data

Known issue(s):
  - Moved all calibrations upstream of the full chain (for the better!)
  - No other known issue


## Configurations for MPV/MPR v04

These weights have been trained/validated using the following files:
- Training set: `/sdf/data/neutrino/icarus/sim/mpvmpr_v4/train_file_list.txt`
- Test set: `/sdf/data/neutrino/icarus/sim/mpvmpr_v4/test_file_list.txt`

This dataset is a superset of v03, with the additional of samples with different
electron lifetimes (4 ms and 8 ms) to match what was recorded during ICARUS run 2.

### June 25th 2025 

```shell
icarus_full_chain_co_250625.cfg
icarus_full_chain_co_lite_250625.cfg
icarus_full_chain_co_numi_250625.cfg
icarus_full_chain_co_4ms_lite_250625.cfg
icarus_full_chain_co_8ms_lite_250625.cfg
icarus_full_chain_co_transp_lite_250625.cfg
icarus_full_chain_data_co_250625.cfg
icarus_full_chain_data_co_lite_250625.cfg
icarus_full_chain_data_co_lowbs_lite_250625.cfg
icarus_full_chain_data_co_unblind_lite_250625.cfg
icarus_full_chain_data_co_numi_250625.cfg
icarus_full_chain_single_co_250625.cfg
```

Description:
  - UResNet + PPN + gSPICE + GrapPAs (track + shower + interaction)
  - Class-weighted loss on PID predictions
  - The `*_co_*` declinations only use collection charge
  - The `*_numi_*` declinations are tailored for NuMI (10 us beam window for FM)
  - The `*_4ms_*` declinations are tailored for 4ms lifetime variations
  - The `*_8ms_*` declinations are tailored for 8ms lifetime variations
  - The `*_transp_*` declinations are tailored for the YZ transparency variations
  - The `*_data_*` declinations are tailored for data (no labels)
  - The `*_lite_*` declinations output directly to lite files
  - The `*_lowbs_*` declinations with low batch size to reduce memory consumption (1)
  - The `*_unblind_*` declination only processes unblinded data
  - The `*_single_*` declination works on single-cryostat simulations

Changes:
  - Weights trained on a mixture of lifetimes (3, 4 and 8 ms)
  - Multiple shower quality cuts added (for nue analysis)
    - Cut on Michel KE of 100 MeV (anything above that is not a Michel)
    - Shower start merge (merge start with track stubs, if present)
    - Shower start point correction (based on vertex distance)
    - Exclude EM activity from containment checks
    - 50 MeV visibility threshold on electron showers (for topology)
    - Compute start dE/dx
    - Compute start straightness
    - Compute particle spread
    - Compute EM shower conversion distance
  - Use true energy depoisitions (SED) for the contaiment check
  - Use the vertex to update the track orientations (fix track flipping)
