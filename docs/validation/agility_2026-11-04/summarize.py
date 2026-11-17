"""Reproduce the finite agility comparison from its JSON and 20ms NPZ traces."""
import importlib.util
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'docs/validation/agility_2026-11-04'
spec = importlib.util.spec_from_file_location('agility_metrics', ROOT / 'scripts/cusrl/evaluate_agility.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def distribution(values):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return {'n': len(values), 'median': float(np.median(values)) if values else None,
            'p90': float(np.quantile(values, .9)) if values else None,
            'mean': float(np.mean(values)) if values else None}


def responses(records, key):
    items = [record[key] for record in records if record[key]['status'] != 'not_applicable']
    return {'statuses': dict(Counter(item['status'] for item in items)), 'total': len(items),
            'latency_s': distribution([item['latency_s'] for item in items]),
            'confirmed_s': distribution([item['confirmed_s'] for item in items]),
            'distance_m': distribution([item.get('stop_distance_xy_m') for item in items]),
            'path_m': distribution([item.get('stop_path_xy_m') for item in items])}


def main():
    partial = '--partial' in sys.argv
    collected, traces = {}, {}
    for policy in ('stage1', 'impact'):
        collected[policy], traces[policy] = {}, {}
        for mode in ('off', 'common'):
            loaded, data = [], []
            for seed in (42, 43):
                path = OUT / f'{policy}_{mode}_seed{seed}.json'
                if not path.exists():
                    if partial:
                        continue
                    raise FileNotFoundError(path)
                report = json.loads(path.read_text())
                trace = dict(np.load(path.with_suffix('.npz')))
                # Independent stop criterion prevents cancellation of signed
                # oscillating velocities before testing the stopping threshold.
                for trial in report['per_trial']:
                    index = trial['env_index']
                    measurements = module.strict_stop_measurements(
                        trace['command_tactical'][:, index], trace['actual_velocity_tactical_raw'][:, index],
                        trace['root_pos_w'][:, index],
                        trace['alive'][:, index], trace['first_fall'][:, index],
                        trace['initial_root_pos_w'][index], report['control_timestep_s'],
                    )
                    for transition, check in zip(trial['transitions'], measurements, strict=True):
                        transition['stop_raw_speed'] = check
                loaded.append(report)
                data.append(trace)
            if not loaded:
                continue
            trials = [trial for report in loaded for trial in report['per_trial']]
            samples = sum(report['aggregate']['valid_samples'] for report in loaded)
            summary = {
                'seeds': [report['seed'] for report in loaded],
                'checkpoint': loaded[0]['checkpoint'], 'trials': len(trials),
                'falls': sum(trial['fell'] for trial in trials), 'valid_samples': samples,
                'tracking_rmse_xy': float(np.sqrt(sum(
                    report['aggregate']['tracking_rmse_xy']**2 * report['aggregate']['valid_samples']
                    for report in loaded) / samples)),
                'tracking_rmse_yaw': float(np.sqrt(sum(
                    report['aggregate']['tracking_rmse_yaw']**2 * report['aggregate']['valid_samples']
                    for report in loaded) / samples)),
                'scenarios': {},
            }
            for name, config in loaded[0]['scenarios'].items():
                selected = [trial for trial in trials if trial['scenario'] == name]
                transitions = [transition for trial in selected for transition in trial['transitions']]
                axis = config['axis']
                scene = {'trials': len(selected), 'falls': sum(trial['fell'] for trial in selected),
                         'reach_50': responses(transitions, 'reach_50'),
                         'reach_90': responses(transitions, 'reach_90'),
                         'zero_cross': responses(transitions, 'zero_cross'),
                         'stop_filtered': responses(transitions, 'stop'),
                         'stop_raw_speed': responses(transitions, 'stop_raw_speed'),
                         'directions': {}}
                if axis is not None:
                    for sign in (1, -1):
                        directed = [tr for tr in transitions if tr['command_after'][axis]*sign > 1e-8]
                        scene['directions'][str(sign)] = {
                            'reach_50': responses(directed, 'reach_50'),
                            'reach_90': responses(directed, 'reach_90'),
                            'steady_gain': distribution([tr['steady_state_gain']['value'] for tr in directed]),
                        }
                summary['scenarios'][name] = scene
            collected[policy][mode] = summary
            traces[policy][mode] = data
            print(policy, mode, 'n=', summary['trials'], 'falls=', summary['falls'],
                  'xyRMSE=', round(summary['tracking_rmse_xy'], 3),
                  'yawRMSE=', round(summary['tracking_rmse_yaw'], 3))
            for name in ('fb_1', 'lr_1', 'yaw_0.75', 'yaw_0.75_negative_first', 'yaw_1.25',
                         'fast_fb', 'fast_lr', 'fast_yaw', 'mixed_chase_evade'):
                scene = summary['scenarios'][name]
                print(' ', name, 'falls', scene['falls'], 'T90', scene['reach_90']['statuses'],
                      'median', scene['reach_90']['latency_s']['median'],
                      'strict stop', scene['stop_raw_speed']['statuses'],
                      'filtered stop', scene['stop_filtered']['latency_s']['median'],
                      'dist', scene['stop_filtered']['distance_m']['median'])
                if name in ('fb_1', 'lr_1', 'yaw_0.75', 'yaw_0.75_negative_first', 'yaw_1.25'):
                    for sign, item in scene['directions'].items():
                        print('   ', sign, 'T50', item['reach_50']['statuses'], item['reach_50']['latency_s']['median'],
                              'T90', item['reach_90']['statuses'], item['reach_90']['latency_s']['median'],
                              'gain', item['steady_gain']['median'])
    if partial:
        return
    result = {'schema_version': 1, 'job_id': 13685,
              'stop_notes': 'stop_filtered: signed velocity causal mean threshold; stop_raw_speed: '
                            'causal mean of raw speed norm and absolute yaw rate, held 100ms; '
                            'conditional medians always accompany status counts.',
              'policies': collected}
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    scenes = [('fb_1', 0, 'Forward/backward: 1 m/s'), ('lr_1', 1, 'Left/right: 1 m/s'),
              ('yaw_0.75_negative_first', 2, 'Right turn first: 0.75 rad/s'),
              ('fast_fb', 0, 'Forward/back reversal every 0.5s'),
              ('fast_lr', 1, 'Left/right reversal every 0.5s'),
              ('fast_yaw', 2, 'Turn reversal every 0.5s')]
    for axis_plot, (name, component, title) in zip(axes.flat, scenes, strict=True):
        for policy, color in (('stage1', '#2474b5'), ('impact', '#d77520')):
            runs = []
            for trace in traces[policy]['off']:
                scene_index = list(trace['scenario_names']).index(name)
                indices = np.flatnonzero(trace['scenario_index'] == scene_index)
                runs.append(trace['actual_velocity_tactical_causal_100ms'][:, indices, component])
            merged = np.concatenate(runs, axis=1)
            time = trace['state_time_s']
            axis_plot.plot(time, np.nanmedian(merged, axis=1), color=color,
                           label='Stage 1 / 85500' if policy == 'stage1' else 'Impact / 115500')
            low, high = np.nanquantile(merged, [.25, .75], axis=1)
            axis_plot.fill_between(time, low, high, alpha=.16, color=color, linewidth=0)
        axis_plot.step(trace['command_time_s'], trace['command_tactical'][:, indices[0], component],
                       where='post', color='#333333', linestyle='--', linewidth=1, label='Command')
        axis_plot.set_title(title)
        axis_plot.set_ylabel('rad/s' if component == 2 else 'm/s')
        axis_plot.grid(alpha=.2)
        axis_plot.set_xlim(0, 15)
    axes[0, 0].legend(loc='lower right', fontsize=8)
    for axis_plot in axes[-1]:
        axis_plot.set_xlabel('Time (s)')
    fig.suptitle('T800 dynamic agility / no impacts / 8 trials per scenario\n'
                 'Median and 25-75% band; causal 100ms velocity; first-fall censoring')
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(OUT / 'agility_response.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
