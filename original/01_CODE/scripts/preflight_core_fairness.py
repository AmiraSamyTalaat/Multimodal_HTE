from pathlib import Path
import argparse, json, sys
import numpy as np
import yaml


def mlp_regressor_params(d_in: int, hidden: int = 64) -> int:
    # Mirrors cm_osr.experts.MLPRegressor exactly:
    # Linear(d,h) + LayerNorm(h) + Linear(h,h) + Linear(h,1)
    return int(d_in * hidden + hidden * hidden + 5 * hidden + 1)


def support_router_params(n_experts: int = 4, features_per_expert: int = 4, hidden: int = 32) -> int:
    # Mirrors cm_osr.router.SupportRouter exactly:
    # input = n_experts*features_per_expert + 2
    d_in = n_experts * features_per_expert + 2
    return int((d_in * hidden + hidden) + (hidden * n_experts + n_experts))


def early_fusion_params(d_in: int, hidden: int) -> int:
    # Mirrors EarlyFusionDRNet: Linear(d,h) + Linear(h,h) + Linear(h,1)
    return int(d_in * hidden + hidden * hidden + 3 * hidden + 1)


def attention_params(dims: dict, d_model: int) -> int:
    # Mirrors GenericAttentionDR exactly.
    total_d = int(sum(dims.values()))
    m = int(len(dims))
    d = int(d_model)
    # modality encoders + score + two-layer head
    return int(d * d + d * (total_d + m + 3) + 2)


def moe_params(dims: dict, hidden: int) -> int:
    # Mirrors GenericMoEDR exactly.
    total_d = int(sum(dims.values()))
    m = int(len(dims))
    h = int(hidden)
    # modality experts + joint router
    return int(h * (2 * total_d + 3 * m + 1) + 2 * m)


def best_size(fn, target: int, values):
    best = None
    for v in values:
        n = int(fn(v))
        err = abs(n - target) / max(target, 1)
        candidate = (err, int(v), n)
        if best is None or candidate < best:
            best = candidate
    return best[1], best[2], best[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    a = p.parse_args()
    root = Path(a.root)

    data = root / '02_DATA' / 'processed' / 'ispy2_core_multimodal.npz'
    man = root / '02_DATA' / 'manifests' / 'CORE_DATA_MANIFEST.json'
    if not data.exists() or not man.exists():
        raise SystemExit('Missing processed core data or manifest. Run Stage 2 first.')

    mj = json.loads(man.read_text(encoding='utf-8'))
    if mj.get('status') != 'PASS':
        raise SystemExit('CORE_DATA_MANIFEST is not PASS.')

    z = np.load(data, allow_pickle=False)
    required = ['clinical', 'transcriptomic', 'rppa']
    missing = [k for k in required if k not in z.files]
    if missing:
        raise SystemExit(f'Processed NPZ is missing modalities: {missing}')

    dims = {k: int(z[k].shape[1]) for k in required}
    ns = {k: int(z[k].shape[0]) for k in required}
    if len(set(ns.values())) != 1:
        raise SystemExit(f'Modality row-count mismatch: {ns}')
    n = next(iter(ns.values()))

    cfg = yaml.safe_load((root / '00_CONFIG' / 'project.yaml').read_text(encoding='utf-8'))
    tol = float(cfg['capacity_tolerance_percent']) / 100.0

    # CM-OSR has one expert per modality plus one joint expert.
    expert_dims = list(dims.values()) + [sum(dims.values())]
    cm_experts = sum(mlp_regressor_params(d, 64) for d in expert_dims)
    cm_router = support_router_params(4, 4, 32)
    cm = int(cm_experts + cm_router)

    d_total = int(sum(dims.values()))
    eh, en, ee = best_size(lambda h: early_fusion_params(d_total, h), cm, range(8, 1025, 8))
    ad, an, ae = best_size(lambda d: attention_params(dims, d), cm, range(8, 513, 4))
    mh, mn, me = best_size(lambda h: moe_params(dims, h), cm, range(8, 513, 4))

    out = {
        'status': 'PASS',
        'scientific_training_started': False,
        'local_torch_required_for_preflight': False,
        'counting_method': 'analytic parameter count mirroring the locked PyTorch module definitions; no model is instantiated or trained locally',
        'n_patients': n,
        'dims': dims,
        'CM_OSR_neural_trainable_params': cm,
        'CM_OSR_components': {
            'four_effect_experts_params': int(cm_experts),
            'support_router_params': int(cm_router),
        },
        'capacity_tolerance': tol,
        'matches': {
            'early_fusion': {'hidden': eh, 'params': en, 'rel_error': ee},
            'generic_attention': {'d_model': ad, 'params': an, 'rel_error': ae},
            'generic_MoE': {'hidden': mh, 'params': mn, 'rel_error': me},
        },
        'training_budget': {
            'max_epochs': int(cfg['training']['max_epochs']),
            'patience_CM_OSR_experts': int(cfg['training']['patience']),
            'same_max_epochs_neural_baselines': True,
        },
        'note': (
            'This is a non-training preflight. Capacity matching is computed analytically from the exact locked layer definitions. '
            'PyTorch is required only in the Kaggle scientific runtime, not on this Windows machine for this gate. '
            'Non-neural density/reliability estimators in CM-OSR are reported separately rather than hidden.'
        ),
    }

    if any(v['rel_error'] > tol + 1e-12 for v in out['matches'].values()):
        out['status'] = 'FAIL'

    od = root / '03_STATE' / 'CORE_PREFLIGHT_FAIRNESS.json'
    od.parent.mkdir(parents=True, exist_ok=True)
    od.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))

    if out['status'] != 'PASS':
        raise SystemExit('CORE PREFLIGHT FAIRNESS FAIL')
    print('CORE PREFLIGHT FAIRNESS PASS. No scientific model has been trained.')


if __name__ == '__main__':
    main()
