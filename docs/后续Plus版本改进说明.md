# AnkleStable-1 后续改进及证据边界

`T800_base` 采用原 `LowCylinders-MN-StdTarget-Plus-AnkleStable-1`。原项目随后还有一个明确的后续任务：

```text
LeoLab-Locomotion-T800-AMP-Terrain-LowCylinders-MN-StdTarget-Plus-AnkleStable-1-Plus
```

此处最后的 `-Plus` 是在原六项踝稳定奖励之上追加的边界约束试验，与地形名称中的第一个 `Plus` 不同。重建参照源提交为 `75c84ceb8416f56316b961300b018d733a8f6605`。新工程只保留用户指定的原级别 1，不注册、不包含这个后续变体。

## 改了什么

新增四项全部只作用于四个踝关节：左踝 pitch `J04_ANKLE_PITCH_L`、左踝 roll `J05_ANKLE_ROLL_L`、右踝 pitch `J10_ANKLE_PITCH_R`、右踝 roll `J11_ANKLE_ROLL_R`。配置使用 `_ANKLE_CFG`，奖励实现也按这四个名称解析 joint IDs。原任务最终配置中已有的 `dof_pos_limits` 奖励也只选中这四个踝关节；机器人模型中各关节自身的物理限位仍沿用原配置。这里新增的是踝部奖励惩罚，不能当作保证关节绝不越界的硬约束。

后续配置继承原 `AnkleStable-1`，保留原六项踝奖励的函数、权重和参数，另加四项独立记录的奖励：

| 新增项 | 完整权重 | 计算含义 |
|---|---:|---|
| `ankle_plus_actual_soft_limit_l2` | -1.0 | 实际角度超软限位的距离，以 **0.05 rad** 归一化后平方；原六项使用完整软限位跨度归一化，这里对小越界更敏感。 |
| `ankle_plus_actual_hard_limit_l2` | -0.20 | 实际角度超模型硬限位的距离，先减去 **1e-4 rad** 数值容差，再取正部，以 **0.02 rad** 归一化后平方。 |
| `ankle_plus_outward_velocity_l2` | -0.02 | 仅在软限位之外惩罚继续向外的速度，以 **2 rad/s** 归一化；靠近硬限位时权重渐增，达到硬限位后饱和为 1；向内返回不受此项惩罚。 |
| `ankle_plus_target_soft_limit_l2` | -0.10 | 复用原 `ankle_target_soft_limit_hinge_l2`，对 action 缩放和零位偏置后的目标越软限位再加一项惩罚。 |

前三个新函数将左右 roll 的**平方误差**权重乘 2，pitch 保持 1；四关节的平均分母保持不变。实现通过给 roll 输入乘 `sqrt(2)` 达到这个效果，因此不是把 roll 角度整体乘 2 后形成四倍误差。第四项复用原函数，没有新增 roll×2。

原六项继续为：目标速度 -0.010，目标加速度 -0.002，实际超速 -0.025，实际加速度 -0.001，目标软限位 -0.20，实际软限位 -0.20。新增 target 项达到完整权重后，与原目标软限位项在同一函数上合计 -0.30；原来的 -0.20 没有被替换。

这次只改奖励，不改动作缩放、guard 零位、关节位置目标、PD 参数、关节限位、轨迹数据或地形。**没有增加动作/位置目标 hard-limit clamp**。`actual_hard_limit` 是越界后的惩罚信号，不是强制把实际关节夹回范围的控制器。1e-4 rad 容差只影响这一奖励，不扩大模型限位，也不代表硬件允许值。

## 9 月 7 日实际训练到了哪里

| 项目 | 可核实记录 |
|---|---|
| Run | `2026-09-07-13-04-37:t800_lowcylinders_ankle_stable_1_plus_seed42` |
| 载入 | 9 月 6 日 AnkleStable-1 run 的 `ckpt_109000.pt`，不是后来保存的 133000。 |
| 计划终点 | 149000，即计划新增 40000 updates。 |
| 并行 | 4 GPU，全局 4096，seed=42。 |
| LR | `finetune_lr_scale=1.0`，保留恢复后的学习率 cap，约 `1.985225621e-5`，没有再次减半。 |
| warmup | `reward_prefix=ankle_plus_`，只有四个新增奖励从 20% 升向 100%，计划持续 2000 updates；原六项已是完整权重。 |
| 输出末尾 | `Iteration 109281 / 149000`，即只进行到起点之后约 281 updates。 |
| 中断 | Slurm `.err` 记录 job 9766 在 `2026-09-07T13:14:04` 被取消，并向各 rank 发 SIGTERM。 |
| 留存权重 | 此 run 的 `ckpt/` 仅有 `ckpt_109000.pt`；没有 110000 或其他训练后的 checkpoint。 |

训练远未完成计划的 warmup，也未达到下一次每 1000 updates 的保存点。现存 `ckpt_109000.pt` 是起始加载点的保存，不能把它当作已吸收新增边界奖励的训练结果。仅凭任务名、奖励设计或末尾的训练 reward，**不能断言后续 Plus 比原 AnkleStable-1 更好**；当前证据不足以做效果比较。取消原因除调度日志所写的 cancellation 之外不作推断。

可核实文件：

- [后续配置](/srv/shared/home/xjh/workspace/zq_leo/source/leo_lab/leo_lab/tasks/locomotion/config/t800_amp/ankle_stable_plus_env_cfg.py) 和 [奖励实现](/srv/shared/home/xjh/workspace/zq_leo/source/leo_lab/leo_lab/tasks/locomotion/config/t800_amp/ankle_stable_plus_mdp.py)。
- [训练 stdout](/srv/shared/home/xjh/workspace/zq_leo/logs/t800_ankle_stable_1_plus_4gpu-9766.out)、[取消记录 stderr](/srv/shared/home/xjh/workspace/zq_leo/logs/t800_ankle_stable_1_plus_4gpu-9766.err)。
- 原 run 的 `info/ankle_finetune.json`：验证 actor/critic/normalization/AMP/optimizer 恢复，记录学习率、warmup 和四项完整权重。
- [现存四卡启动脚本](/srv/shared/home/xjh/workspace/zq_leo/t800_amp_ankle_stable_1_plus_resume_4gpu.slurm)。

## 原训练命令原文

下面复制该 run 的 `info/command.txt`。完整外层四卡启动命令同时写在上述 stdout 开头，为 `python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 scripts/cusrl/train.py ...`。

```bash
/srv/shared/home/xjh/app/miniconda3/envs/leo/bin/python -u scripts/cusrl/train.py --task=LeoLab-Locomotion-T800-AMP-Terrain-LowCylinders-MN-StdTarget-Plus-AnkleStable-1-Plus --checkpoint=logs/cusrl/t800_base_amp/2026-09-06-19-30-18:t800_lowcylinders_ankle_stable_1_seed42/ckpt/ckpt_109000.pt --max_iterations=149000 --num_envs=4096 --seed=42 --save_interval=1000 --finetune_lr_scale=1.0 --finetune_reward_prefix=ankle_plus_ --finetune_warmup_iterations=2000 --finetune_origin_iteration=109000 --headless --no-debug-vis --run_name=t800_lowcylinders_ankle_stable_1_plus_seed42
```
