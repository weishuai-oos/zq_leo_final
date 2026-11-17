# T800_base

仅训练 T800 的走路 AMP 工程，来源为原工程的
`LeoLab-Locomotion-T800-AMP-Terrain-LowCylinders-MN-StdTarget-Plus-AnkleStable-1`。
第一阶段的有效配置已展开；第二阶段在其 checkpoint 上继续训练抗击打能力。

- `T800_base`：从零训练，31 项奖励中包括原 AnkleStable-1 的六项脚踝奖励。
- `T800_base-Play`：同一策略的评估配置，关闭随机推力。
- `T800_base-Impact` / `T800_base-Impact-Play`：抗击打续训与评估，网络维度和第一阶段一致。
- `T800_base-ImpactV2` / `T800_base-ImpactV2-Play`：从Combined V2续训抗击打，保留Kd2.0、2毫米接触偏移和物理子步防抖。
- 地形：平地 20%、Perlin 8%、粗糙地 12%、波浪 12%、低箱 8%、水平障碍 40%。
  采用 10 行 × 50 列，精确表示每种地形的比例。

[任务来源、最终配置及历史续训链](docs/任务来源与最终配置.md)；
[后续 AnkleStable-1-Plus 改进了什么](docs/后续Plus版本改进说明.md)。

第二阶段参数、启动方式和验证边界见 [抗击打续训说明](docs/抗击打续训说明.md)。

## 环境

沿用原工程已配好的 IsaacLab / Isaac Sim / CusRL 环境；无需安装原 `leo_lab`。
本机可用解释器为 `/srv/shared/home/xjh/app/miniconda3/envs/leo/bin/python`，
版本是 Python 3.11、IsaacLab 2.3.2、Isaac Sim 5.1、CusRL 1.1.8.post0、Torch 2.7。
实际训练需要可用的 NVIDIA GPU。

```bash
cd /srv/shared/home/xjh/workspace/zq_leo_final
conda activate leo
python scripts/list_envs.py
```

入口会自动加入当前工程的源码路径，可以直接运行。若其他程序需要导入本包，可选安装：

```bash
python -m pip install -e . --no-deps
```

包名为 `t800_lab`，T800 的 URDF、网格、14 份动作 NPZ 和采样列表均在本工程内。
URDF 转换在各进程自己的临时目录中进行，避免多卡争用及旧安装的缓存路径。

## 从零训练

单卡：

```bash
python scripts/cusrl/train.py --task T800_base --headless --no-debug-vis \
  --num_envs 4096 --seed 42 --max_iterations 133000 --run_name T800_base_seed42
```

四卡（`--num_envs` 是所有卡的总环境数，各卡 1024）：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/cusrl/train.py --task T800_base --headless --no-debug-vis \
  --num_envs 4096 --seed 42 --max_iterations 133000 --run_name T800_base_4gpu_seed42
```

默认不传 `--checkpoint`，PPO、AMP 判别器及归一化统计均重新初始化。
初始学习率 `1e-3`，六项脚踝奖励从第一步采用最终权重；没有旧续训专用的降学习率或奖励渐增。
保留地形课程，从低难度开始自动升降级，训练过程中不切换任务。

`133000` 是参考历史最终更新编号设置的默认训练预算，不代表新地形从零训练已达到原策略效果。
每 1000 次更新保存，日志在 `logs/cusrl/T800_base/`。集群有时限时仍可用同一任务正常恢复，
不需要重走历史多个任务：

```bash
python scripts/cusrl/train.py --headless --no-debug-vis --num_envs 4096 \
  --checkpoint /absolute/path/to/ckpt_40000.pt --max_iterations 133000
```

`--max_iterations` 为训练的最终总更新数，恢复时应大于 checkpoint 已完成的更新数。
Slurm 脚本已按当前集群配置，可直接提交：

```bash
cd /srv/shared/home/xjh/workspace/zq_leo_final
bash scripts/train.slurm --dry-run   # 仅打印检查和训练命令
mkdir -p logs/slurm
sbatch scripts/train.slurm
```

脚本默认申请 `rtx3090` 分区单节点 2 张 RTX 3090、16 个 CPU、48 GB 内存，时限 4 天。
脚本覆盖通用 agent 默认预算：训练到 100000 次更新，每 500 次保存。
4 天是最长运行时限，达到更新预算后会自行结束，不代表预计耗时或收敛保证。
脚本固定新工程目录和上述 Leo Python，不依赖提交前是否激活 Conda。

作业启动依次检查 CUDA、运行 2 个环境 / 4 步真实 IsaacLab 冒烟，然后开始双卡训练。
任何检查失败都会退出；冒烟不保存策略，不计入训练更新。默认从零训练，
只有显式设置 `CHECKPOINT` 才恢复；例如：

```bash
sbatch --export=ALL,CHECKPOINT=/absolute/path/to/ckpt_40000.pt scripts/train.slurm
```

可用环境变量覆盖 `NUM_ENVS`、`SEED`、`MAX_ITERATIONS`、`SAVE_INTERVAL` 和 `RUN_NAME`。
`--num_envs` 始终是全局数量，默认 4096，即各卡 2048；`SAVE_INTERVAL` 默认 500。
Slurm 输出为 `logs/slurm/T800_base-<jobid>.out/.err`，其中会打印实际启动命令；
提交前需创建 `logs/slurm/`，Slurm 会在执行脚本正文前打开输出文件。
TensorBoard 日志和 checkpoint 保存在 `logs/cusrl/T800_base/` 下的各次运行目录。

## Play、记录及导出

使用本工程训练出的 checkpoint；显式指定地形类型，避免只指定难度却误测其他地形：

```bash
python scripts/cusrl/play.py --load_run /absolute/path/to/ckpt_133000.pt \
  --num_envs 4 --terrain_level 9 --terrain_type fight_mat_horizontal_obstacles \
  --auto_command_test boxing_stability_sweep --record_dir play_records --no-debug-vis
```

自动命令测试完成后退出。无图形界面可加 `--headless`；普通交互 Play 可用 `--teleop keyboard`。
平地选择 `--terrain_level 0 --terrain_type fight_mat_flat`，其他地形名见 `terrain_cfg.py`。
诊断包含脚踝目标/实际位置速度、限位、接触、命令与 episode 统计。

```bash
python scripts/cusrl/export.py --load_run /absolute/path/to/ckpt_133000.pt --format onnx
```

## 检查

```bash
python -m pytest -q
python scripts/check_config.py --headless
python scripts/check_config.py --headless --smoke
```

CPU 回归包含新旧有效配置对照、观测顺序和维度、脚踝奖励行为、地形几何、AMP 数据和 URDF 引用。
受控配置比较不能替代真实物理仿真；`--smoke` 在可用 GPU 上构造少量环境并步进。
本次实际执行的验证及限制见 [验证记录](docs/验证记录.md)。

## 主要文件

| 文件 | 用途 |
|---|---|
| `source/t800_lab/t800_lab/tasks/t800_base/env_cfg.py` | 直接定义场景、观测、命令、31 项奖励、事件及课程 |
| `source/t800_lab/t800_lab/tasks/t800_base/terrain_cfg.py` | 六种地形的比例和参数 |
| `source/t800_lab/t800_lab/tasks/t800_base/agent_cfg.py` | LSTM PPO、AMP 和训练预算 |
| `source/t800_lab/t800_lab/tasks/t800_base/constants.py` | 关节顺序、AMP 掩码及姿态目标 |
| `source/t800_lab/t800_lab/tasks/t800_base/ankle_stable_mdp.py` | 六项脚踝奖励及差分缓存 |

仅保留本任务实际使用的运行时、遥操作和诊断工具，不复制原日志或 checkpoint。
原项目的授权文本保存在 `LICENSE`，原代码中的授权声明保留。
