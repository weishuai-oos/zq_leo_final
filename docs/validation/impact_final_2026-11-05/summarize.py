"""Reproduce the final-checkpoint validation and historical protocol references."""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
HISTORICAL = ROOT / 'docs/validation/impact_comparison_2026-11-04'
AGILITY = ROOT / 'docs/validation/agility_2026-11-04'


def load(path):
    return json.loads(path.read_text())


def distribution(values):
    values = [value for value in values if value is not None]
    return {'n': len(values), 'median': float(np.median(values)) if values else None,
            'mean': float(np.mean(values)) if values else None}


def aggregate(items):
    trials = sum(item['trials'] for item in items)
    samples = sum(item['valid_samples'] for item in items)
    recovery_count = sum(item['impacts']['recoveries'] for item in items)
    return {
        'trials': trials, 'falls': sum(item['falls'] for item in items),
        'alive_seconds_mean': sum(item['alive_seconds_total'] for item in items) / trials,
        'valid_samples': samples,
        'tracking_rmse_xy': float(np.sqrt(sum(item['tracking_rmse_xy']**2 * item['valid_samples']
                                              for item in items) / samples)),
        'tracking_rmse_yaw': float(np.sqrt(sum(item['tracking_rmse_yaw']**2 * item['valid_samples']
                                               for item in items) / samples)),
        'actual_velocity_mean_tactical': (sum(np.asarray(item['actual_velocity_mean_tactical']) *
                                              item['valid_samples'] for item in items) / samples).tolist(),
        'impacts': {
            'hits': sum(item['impacts']['hits'] for item in items),
            'recoveries': recovery_count,
            'recovery_expired': sum(item['impacts']['recovery_expired'] for item in items),
            'completed_recovery_mean_s': sum((item['impacts']['completed_recovery_mean_s'] or 0) *
                                              item['impacts']['recoveries'] for item in items) / recovery_count
                                         if recovery_count else None,
            'extra_displacement_max_m': max(item['impacts']['extra_displacement_max_m'] for item in items),
        },
    }


def fixed_protocol(folder, prefix):
    modes = {}
    for mode in ('off', 'common', 'stress'):
        reports = [load(folder / f'{prefix}_{mode}_seed{seed}.json') for seed in (42, 43)]
        modes[mode] = {
            'checkpoint_iteration': reports[0]['checkpoint_iteration'],
            **aggregate([report['aggregate'] for report in reports]),
            'commands': {name: aggregate([report['commands'][name] for report in reports])
                         for name in reports[0]['commands']},
        }
    return modes


def agility_summary(folder, prefix):
    reports = [load(folder / f'{prefix}_off_seed{seed}.json') for seed in (42, 43)]
    trials = [trial for report in reports for trial in report['per_trial']]
    samples = sum(report['aggregate']['valid_samples'] for report in reports)
    scenes = {}
    for name, scenario in reports[0]['scenarios'].items():
        selected = [trial for trial in trials if trial['scenario'] == name]
        transitions = [transition for trial in selected for transition in trial['transitions']]
        axis = scenario['axis']
        result = {'trials': len(selected), 'falls': sum(trial['fell'] for trial in selected), 'directions': {}}
        for key in ('reach_50', 'reach_90', 'zero_cross', 'stop_raw_speed'):
            entries = [transition[key] for transition in transitions if transition[key]['status'] != 'not_applicable']
            result[key] = {'windows': len(entries), 'reached': sum(item['status'] == 'reached' for item in entries),
                           'latency_s': distribution([item['latency_s'] for item in entries])}
        if axis is not None:
            for sign in (1, -1):
                edges = [edge for edge in transitions if edge['command_after'][axis]*sign > 1e-8]
                result['directions'][str(sign)] = {
                    'windows': len(edges),
                    'reach_50': sum(edge['reach_50']['status'] == 'reached' for edge in edges),
                    'reach_90': sum(edge['reach_90']['status'] == 'reached' for edge in edges),
                    'reach_90_latency_s': distribution([edge['reach_90']['latency_s'] for edge in edges]),
                    'steady_gain': distribution([edge['steady_state_gain']['value'] for edge in edges]),
                }
        scenes[name] = result
    negative = [edge for trial in trials if trial['scenario'] in ('yaw_0.75', 'yaw_0.75_negative_first')
                for edge in trial['transitions'] if edge['command_after'][2] < 0]
    return {
        'checkpoint_iteration': reports[0]['checkpoint_iteration'],
        'trials': len(trials), 'falls': sum(trial['fell'] for trial in trials),
        'tracking_rmse_xy': float(np.sqrt(sum(report['aggregate']['tracking_rmse_xy']**2 *
                                              report['aggregate']['valid_samples'] for report in reports) / samples)),
        'tracking_rmse_yaw': float(np.sqrt(sum(report['aggregate']['tracking_rmse_yaw']**2 *
                                               report['aggregate']['valid_samples'] for report in reports) / samples)),
        'low_right_turn': {
            'windows': len(negative), 'reach_90': sum(edge['reach_90']['status'] == 'reached' for edge in negative),
            'final_0_5s_actual_rate_rad_s': distribution(
                [-0.75 * edge['steady_state_gain']['value'] for edge in negative
                 if edge['steady_state_gain']['value'] is not None]),
        },
        'scenarios': scenes,
    }


def plot_result(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    groups = [
        ('Stage 1 / 85500', result['historical_fixed_protocol']['stage1'], '#3976b2'),
        ('Impact / 115000', result['historical_fixed_protocol']['impact'], '#dc8b2c'),
        ('Impact / 125500', result['final_fixed_protocol'], '#2b9569'),
    ]
    for index, (label, values, color) in enumerate(groups):
        x = np.arange(2) + (index - 1) * .24
        counts = [values[mode]['falls'] for mode in ('common', 'stress')]
        bars = axes[0].bar(x, counts, width=.23, label=label, color=color)
        axes[0].bar_label(bars, labels=[f'{count}/96' for count in counts], padding=3, fontsize=9)
    axes[0].set_xticks([0, 1], ['Common impacts', 'Stress / isolated hits'])
    axes[0].set_ylabel('Trials ending in first fall')
    axes[0].set_ylim(0, 100)
    axes[0].set_title('Fixed commands / up to 20s / 96 trials per mode')
    axes[0].legend(fontsize=8)
    axes[0].grid(axis='y', alpha=.2)
    for folder, prefix, label, color in (
        (AGILITY, 'impact', 'Impact / 115500', '#dc8b2c'),
        (OUT, 'agility', 'Impact / 125500', '#2b9569'),
    ):
        velocity = []
        for seed in (42, 43):
            with np.load(folder / f'{prefix}_off_seed{seed}.npz') as trace:
                scene = list(trace['scenario_names']).index('yaw_0.75_negative_first')
                selected = np.flatnonzero(trace['scenario_index'] == scene)
                velocity.append(trace['actual_velocity_tactical_causal_100ms'][:, selected, 2])
                time = trace['state_time_s'].copy()
                command_time = trace['command_time_s'].copy()
                command = trace['command_tactical'][:, selected[0], 2].copy()
        values = np.concatenate(velocity, axis=1)
        axes[1].plot(time, np.nanmedian(values, axis=1), color=color, label=label)
        low, high = np.nanquantile(values, [.25, .75], axis=1)
        axes[1].fill_between(time, low, high, color=color, alpha=.15, linewidth=0)
    axes[1].step(command_time, command, where='post', color='#333333', linestyle='--', label='Command')
    axes[1].set_title('Right turn first / 0.75 rad/s / 8 trials')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('World-z angular velocity (rad/s)')
    axes[1].grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle('Final T800 Impact checkpoint validation')
    fig.text(.5, .015, 'Descriptive test counts, not match probabilities. Historical: RTX3090; final: RTX A4000.\n'
             'Turn curves: median and 25-75% band, causal 100ms velocity, first-fall censoring.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .93))
    fig.savefig(OUT / 'final_comparison.png', dpi=170)
    plt.close(fig)


def main():
    result = {
        'job_id': 13742,
        'gpu': '2 x NVIDIA RTX A4000 (tr1)',
        'final_fixed_protocol': fixed_protocol(OUT, 'impact'),
        'historical_fixed_protocol': {
            'stage1': fixed_protocol(HISTORICAL, 'stage1'),
            'impact': fixed_protocol(HISTORICAL, 'impact'),
        },
        'final_agility': agility_summary(OUT, 'agility'),
        'historical_agility': agility_summary(AGILITY, 'impact'),
        'notes': [
            'Each final fixed-command mode has 96 independent 20-second first-fall trials, two seeds, 12 commands.',
            'Stress means isolated single hits per event, with 2-5s intervals; multiple events per 20s trial are possible.',
            'All evaluation trials retain healthy physical heads and body/friction randomization.',
            'Historical constant-command impact checkpoint is 115000, while historical agility checkpoint is 115500.',
            'Historical tests used RTX3090; final tests use RTX A4000. These are historical protocol references, not same-hardware causal ablations.',
            'Recovery duration averages completed recoveries only; expiration and fall counts are separate.',
            'All actual metrics stop at first fall; conditional latency always accompanies reached/window counts.',
        ],
    }
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    plot_result(result)
    print(json.dumps(result['final_fixed_protocol'], indent=2))
    print('Final agility:', json.dumps(result['final_agility'], indent=2))


if __name__ == '__main__':
    main()
