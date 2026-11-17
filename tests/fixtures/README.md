# Historical configuration fixtures

`old_train_config.json` and `old_play_config.json` were extracted read-only from
`../zq_leo/source/leo_lab/leo_lab/tasks/locomotion/config/t800_amp/ankle_stable_env_cfg.py`:

- `T800LowCylinderTerrainAmpMaskBaseNarrowStdTargetPlusAnkleStable1EnvCfg`
- `T800LowCylinderTerrainAmpMaskBaseNarrowStdTargetPlusAnkleStable1EnvCfg_PLAY`

The extractor in `tests/_config_model.py` executes the original configuration
class definitions and every inherited `__post_init__`. External IsaacLab
constructors record their explicit arguments without starting Kit. Local Plus
terrain class defaults are also included. This model does not reproduce
IsaacLab defaults or physics.

`test_config_contract.py` permits only the requested terrain mix and 50 columns,
the task run name, removal of the empty custom monitor configuration and the
unused uppercase level constant, and omission of four disabled (`None`) AMP
observation terms. Function references compare by name; asset and dataset paths
compare relative to the copied resource roots.

For a comparison using real IsaacLab defaults and constructors, run:

```sh
python scripts/check_config.py --headless --compare-original ../zq_leo
```

The optional original-project path is used for verification only. New task
imports and training do not depend on the old project or these fixtures.
