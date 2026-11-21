"""Build a patient-free ZIP from completed, independently retrained Kaggle results."""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
CODE_FILES = ['RUN_FROZEN_INDEPENDENT_RERUN.py','COMPARE_INDEPENDENT_RERUN.py',
              'b01_independent_kernel.py','orchestrate_stage4.py','orchestrate_selected_stages.py',
              'resume_paper1.py','revalidate_saved_aggregate.py','validate_comparator.py','auto_finalize_results.py',
              'RUN_RESUME_PAPER1.cmd','README_REPRODUCTION.md','ENVIRONMENT_COMPARISON.md',
              'fair_neural_template.py','fair_neural_25fit_pilot.py',
              'collect_fair_neural.py','package_actual_results.py']
FAIR_FILES = ['FAIR_NEURAL_COMPARISON.csv','FAIR_NEURAL_SUMMARY.csv','FAIR_NEURAL_PAIRED_DGP.csv',
              'FAIR_NEURAL_TRAINING_BUDGET.csv','FAIR_NEURAL_VALIDATION.json','RESOURCE_PILOT_FINDINGS.json']
SELECTED = {
    'stage6a': ('07_METHOD_DEV_RUNS','STAGE6A/ALL_STAGE6A_DEV_METRICS.csv','STAGE6A/STAGE6A_SELECTION.json',10),
    'stage6b': ('08_CONFIRMATORY_RUNS','STAGE6B/ALL_STAGE6B_TEST_METRICS.csv','STAGE6B/STAGE6B_CONFIRMATORY_DECISION.json',15),
    'stage6c': ('08_MECHANISM_RUNS','STAGE6C/ALL_STAGE6C_MECHANISM_DEV_METRICS.csv','STAGE6C/STAGE6C_MECHANISM_DECISION.json',10),
    'stage7b': ('08_STAGE7_DEV_RUNS','STAGE7B/ALL_STAGE7B_BASELINE_DEV_METRICS.csv','STAGE7B/STAGE7B_DEVELOPMENT_SELECTION.json',12),
}


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def csv_rows(payload):
    return list(csv.DictReader(io.StringIO(payload.decode('utf-8-sig'))))


def csv_bytes(rows):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode('utf-8-sig')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=('gpu','cpu'), default='gpu')
    a = ap.parse_args()
    suffix = '-cpu' if a.backend == 'cpu' else ''
    stage_slug = 'b01-p1-independent-stage4-aggregate' + suffix
    stage_dir = BASE/'downloads'/stage_slug
    required = [stage_dir/'B01_INDEPENDENT_CHECKPOINT.zip', stage_dir/'INDEPENDENT_RERUN_COMPARISON.json',
                stage_dir/'KERNEL_RUN_MANIFEST.json']
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise SystemExit('Full Stage4 aggregation not yet downloaded: '+', '.join(missing))
    local_stage_comparison = stage_dir/'LOCAL_VALIDATED_COMPARISON.json'
    effective_comparison_path = local_stage_comparison if local_stage_comparison.is_file() else required[1]
    comparison = json.loads(effective_comparison_path.read_text(encoding='utf-8'))
    with zipfile.ZipFile(BASE/'B01_Paper1_Complete_Reviewer_Code_Updated.zip') as original_zip:
        archived_gate = json.loads(original_zip.read('original/06_RESULTS/PRIMARY_GATE.json'))
    kernel_manifest = json.loads(required[2].read_text(encoding='utf-8'))
    if sha(required[0].read_bytes()) != kernel_manifest['checkpoint_sha256']:
        raise SystemExit('Stage4 checkpoint SHA-256 mismatch')
    with zipfile.ZipFile(required[0]) as z:
        if z.testzip() is not None:
            raise SystemExit('Stage4 checkpoint CRC failure')
        names = z.namelist()
        if any(n.endswith('.npz') or '/predictions/' in n or 'patient_dev_predictions' in n for n in names):
            raise SystemExit('Stage4 checkpoint contains patient-level content')
        stage_rows = csv_rows(z.read('B01/06_RESULTS/ALL_CORE_METRICS.csv'))
        gate = json.loads(z.read('B01/06_RESULTS/PRIMARY_GATE.json'))
        run_manifests = [n for n in names if n.startswith('B01/04_RUNS/') and n.endswith('/RUN_MANIFEST.json')]
        stage_logs = [n for n in names if n.startswith('progress/') and n.endswith('.log')]
        if len({(r['dgp_seed'],r['net_seed']) for r in stage_rows}) != 15 or len(run_manifests) != 15:
            raise SystemExit('Stage4 grid is not 15 unique pairs with 15 manifests')
        if not gate['grid_complete'] or gate['observed_dgp_clusters'] != 5:
            raise SystemExit('Stage4 scientific gate was not computed on five DGP clusters')
        files = {}
        for n in run_manifests:
            files['RUN_MANIFESTS/STAGE4/'+Path(n).parent.name+'.json'] = z.read(n)
        for n in stage_logs:
            files['RUN_LOGS/STAGE4/'+Path(n).name] = z.read(n)
        for n in names:
            if n.startswith('B01/06_RESULTS/') and n.endswith(('.csv','.json')):
                files['STAGE4_AGGREGATES/'+n[len('B01/06_RESULTS/'):]] = z.read(n)
    selected_rows, selected_validations, comparison_reports = [], {}, {'stage4': comparison}
    files['STAGE_COMPARISONS/stage4.json'] = effective_comparison_path.read_bytes()
    if local_stage_comparison.is_file():
        files['STAGE_COMPARISONS/stage4_original_kaggle_comparator.json'] = required[1].read_bytes()
        files['STAGE_COMPARISONS/stage4_local_validation_provenance.json'] = (stage_dir/'LOCAL_VALIDATED_PROVENANCE.json').read_bytes()
    for selected_stage, (run_folder, table, decision_name, pair_count) in SELECTED.items():
        selected_slug = f'b01-p1-independent-{selected_stage}-aggregate{suffix}'
        selected_dir = BASE/'downloads'/selected_slug
        cp_path = selected_dir/'B01_INDEPENDENT_CHECKPOINT.zip'
        cmp_path = selected_dir/'INDEPENDENT_RERUN_COMPARISON.json'
        man_path = selected_dir/'KERNEL_RUN_MANIFEST.json'
        if not all(p.is_file() for p in (cp_path,cmp_path,man_path)):
            selected_validations[selected_stage] = {'status':'PENDING', 'expected_seed_pairs':pair_count}
            continue
        selected_manifest = json.loads(man_path.read_text(encoding='utf-8'))
        if selected_manifest['runner_returncode'] != 0 or sha(cp_path.read_bytes()) != selected_manifest['checkpoint_sha256']:
            raise SystemExit(f'{selected_stage} checkpoint failed technical validation')
        local_compare_path = selected_dir/'LOCAL_VALIDATED_COMPARISON.json'
        effective_selected_cmp = local_compare_path if local_compare_path.is_file() else cmp_path
        selected_compare = json.loads(effective_selected_cmp.read_text(encoding='utf-8'))
        comparison_reports[selected_stage] = selected_compare
        files['STAGE_COMPARISONS/'+selected_stage+'.json'] = effective_selected_cmp.read_bytes()
        if local_compare_path.is_file():
            files['STAGE_COMPARISONS/'+selected_stage+'_original_kaggle_comparator.json'] = cmp_path.read_bytes()
            files['STAGE_COMPARISONS/'+selected_stage+'_local_validation_provenance.json'] = (selected_dir/'LOCAL_VALIDATED_PROVENANCE.json').read_bytes()
        with zipfile.ZipFile(cp_path) as z:
            if z.testzip() is not None:
                raise SystemExit(f'{selected_stage} checkpoint CRC failure')
            names = z.namelist()
            table_rows = csv_rows(z.read('B01/06_RESULTS/'+table))
            decision = json.loads(z.read('B01/06_RESULTS/'+decision_name))
            with zipfile.ZipFile(BASE/'B01_Paper1_Complete_Reviewer_Code_Updated.zip') as original_zip:
                archived_decision = json.loads(original_zip.read('original/06_RESULTS/'+decision_name))
            manifests = [n for n in names if n.startswith('B01/'+run_folder+'/') and n.endswith('/RUN_MANIFEST.json')]
            if len(manifests) != pair_count:
                raise SystemExit(f'{selected_stage} has {len(manifests)} manifests, expected {pair_count}')
            for n in manifests:
                files['RUN_MANIFESTS/'+selected_stage.upper()+'/'+Path(n).parent.name+'.json'] = z.read(n)
            if selected_stage == 'stage6b':
                dependency = 'progress/stage6b_archived_selection_dependency.json'
                if dependency not in names:
                    raise SystemExit('Stage6B frozen archived candidate dependency is missing')
                files['RUN_MANIFESTS/STAGE6B/frozen_archived_stage6a_selection_dependency.json'] = z.read(dependency)
            for n in names:
                if n.startswith('progress/') and n.endswith('.log'):
                    files['RUN_LOGS/'+selected_stage.upper()+'/'+Path(n).name] = z.read(n)
                if n.startswith('B01/06_RESULTS/'+selected_stage.upper()+'/') and n.endswith(('.csv','.json')):
                    if 'patient' in n.lower() or 'prediction' in n.lower():
                        raise SystemExit(f'{selected_stage} aggregate path may contain patient-level content: {n}')
                    files['STAGE_AGGREGATES/'+selected_stage.upper()+'/'+Path(n).name] = z.read(n)
        for row in table_rows:
            row['experiment'] = 'independent_frozen_'+selected_stage
            row['provenance'] = f'kaggle:prinprin/{selected_slug}'
            row['network_seed'] = row.get('net_seed',row.get('learner_seed',''))
            row['effect_regime'] = row.get('regime',row.get('condition',''))
            row['calibration'] = 'original'
        selected_rows.extend(table_rows)
        selected_validations[selected_stage] = {'status':'COMPLETE','expected_seed_pairs':pair_count,
            'decision_status':decision.get('status'), 'archived_decision_status':archived_decision.get('status'),
            'grid_complete':decision.get('grid_complete'), 'run_utc':selected_manifest['utc'],
            'gates':decision.get('gates'), 'selected_candidate':decision.get('selected_candidate'),
            'categorical_decision_matches_archive': any(c['type']=='scientific_decision' for c in selected_compare['checks']) and all(not c.get('mismatched_decisions') for c in selected_compare['checks'] if c['type']=='scientific_decision'),
            'strict_numerical_comparison_pass':selected_compare['all_comparisons_pass'],
            'strict_numerical_failed_checks':sum(not c['pass'] for c in selected_compare['checks'])}
        for logfile in selected_dir.glob('*.log'):
            files['RUN_LOGS/'+selected_stage.upper()+'_KERNEL/'+logfile.name] = logfile.read_bytes()
    fair_rows = csv_rows((BASE/'FAIR_NEURAL_COMPARISON.csv').read_bytes())
    fair_validation = json.loads((BASE/'FAIR_NEURAL_VALIDATION.json').read_text(encoding='utf-8'))
    if fair_validation['status'] != 'PASS' or len(fair_validation['checks']) != 24:
        raise SystemExit('Fair neural validation incomplete')
    for row in stage_rows:
        row['experiment'] = 'independent_frozen_stage4'
        row['provenance'] = f'kaggle:prinprin/{stage_slug}'
        row['effect_regime'] = 'original_stage4_dgp'
        row['calibration'] = 'original'
        row['network_seed'] = row.get('net_seed','')
    for row in fair_rows:
        row['experiment'] = 'post_review_fair_neural'
    all_rows = stage_rows + selected_rows + fair_rows
    cols = list(dict.fromkeys(k for r in all_rows for k in r))
    files['EXPERIMENT_RESULTS.csv'] = csv_bytes([{k:r.get(k,'') for k in cols} for r in all_rows])
    files['INDEPENDENT_RERUN_COMPARISON.json'] = json.dumps({
        'stages_included':list(comparison_reports),
        'all_comparisons_pass':all(x['all_comparisons_pass'] for x in comparison_reports.values()),
        'reports':comparison_reports}, indent=2).encode()
    for n in FAIR_FILES:
        files[n] = (BASE/n).read_bytes()
    for n in CODE_FILES:
        files['SOURCE_CODE/'+n] = (BASE/n).read_bytes()
    files['SOURCE_CODE/B01_Paper1_Complete_Reviewer_Code_Updated.zip'] = (BASE/'B01_Paper1_Complete_Reviewer_Code_Updated.zip').read_bytes()
    for source in fair_validation['sources']:
        path = BASE/'downloads'/source['kernel'].split('/')[-1]/source['file']
        if sha(path.read_bytes()) != source['sha256']:
            raise SystemExit(f'Fair neural batch SHA-256 mismatch: {path}')
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None or any(x.endswith('.npz') or 'prediction' in x.lower() for x in z.namelist()):
                raise SystemExit(f'Fair neural batch contains prohibited data: {path}')
            for n in z.namelist():
                if n.endswith('/COMPLETE.json'):
                    files['RUN_MANIFESTS/FAIR_NEURAL/'+Path(n).parent.name+'.json'] = z.read(n)
        files['RAW_FAIR_NEURAL_BATCHES/'+path.parent.name+'.zip'] = path.read_bytes()
        for logfile in path.parent.glob('*.log'):
            files['RUN_LOGS/FAIR_NEURAL/'+logfile.name] = logfile.read_bytes()
    pilot = BASE/'downloads'/'fair_smoke_cpu'/'B01_P1_FAIR_NEURAL_BATCH.zip'
    pilot_info = json.loads((BASE/'RESOURCE_PILOT_FINDINGS.json').read_text(encoding='utf-8-sig'))
    if sha(pilot.read_bytes()) != pilot_info['archive_sha256']:
        raise SystemExit('25-fit pilot SHA-256 mismatch')
    files['RAW_RESOURCE_PILOT/B01_P1_FAIR_NEURAL_BATCH.zip'] = pilot.read_bytes()
    for logfile in pilot.parent.glob('*.log'):
        files['RUN_LOGS/RESOURCE_PILOT/'+logfile.name] = logfile.read_bytes()
    for logfile in stage_dir.glob('*.log'):
        files['RUN_LOGS/STAGE4_KERNEL/'+logfile.name] = logfile.read_bytes()
    for logfile in (BASE/'RUN_LOGS').glob('*'):
        files['RUN_LOGS/RETRIES/'+logfile.name] = logfile.read_bytes()
    findings = (BASE/'FINAL_SCIENTIFIC_FINDINGS.md').read_text(encoding='utf-8')
    finding_lines = ['\n## Live full-grid decision status\n',
                     'This section is generated only from downloaded, SHA-validated Kaggle aggregate checkpoints.\n']
    for selected_stage in ('stage6a','stage6b','stage6c'):
        info = selected_validations[selected_stage]
        if info['status'] != 'COMPLETE':
            finding_lines.append(f'- {selected_stage.upper()}: PENDING; no full-grid decision is claimed.')
            continue
        finding_lines.append(f"- {selected_stage.upper()}: archived `{info['archived_decision_status']}`; "
                             f"new `{info['decision_status']}`; categorical match "
                             f"`{info['categorical_decision_matches_archive']}`; strict numeric match "
                             f"`{info['strict_numerical_comparison_pass']}` "
                             f"({info['strict_numerical_failed_checks']} failed checks).")
        if info.get('selected_candidate'):
            finding_lines.append(f"  Frozen/selected candidate in this decision: `{info['selected_candidate']}`.")
        if info.get('gates'):
            finding_lines.append('  Gate results: '+', '.join(f"{k}={v}" for k,v in info['gates'].items())+'.')
    files['FINAL_SCIENTIFIC_FINDINGS.md'] = (findings+'\n'.join(finding_lines)+'\n').encode('utf-8')
    environment = {'stage4': kernel_manifest.get('environment_versions', {}),
                   'python': kernel_manifest.get('python'), 'platform': kernel_manifest.get('platform'),
                   'requirements_source': 'SOURCE_CODE/B01_Paper1_Complete_Reviewer_Code_Updated.zip/original/requirements.txt'}
    files['ENVIRONMENT.json'] = json.dumps(environment, indent=2).encode()
    files['ENVIRONMENT_COMPARISON.md'] = (BASE/'ENVIRONMENT_COMPARISON.md').read_bytes()
    files['COMPARATOR_VALIDATION.json'] = (BASE/'COMPARATOR_VALIDATION.json').read_bytes()
    for archived_manifest in (BASE/'archived_environment_manifests').glob('*.json'):
        files['ARCHIVED_RUN_MANIFESTS/'+archived_manifest.name] = archived_manifest.read_bytes()
    validation = {'created_utc': datetime.now(timezone.utc).isoformat(), 'stage4_backend': a.backend,
                  'stage4_unique_seed_pairs': 15, 'stage4_independent_dgp_seeds': 5,
                  'stage4_primary_gate': gate['primary_gate'], 'stage4_archived_primary_gate':archived_gate['primary_gate'],
                  'stage4_gate_mean': gate['mean'], 'stage4_archived_gate_mean':archived_gate['mean'],
                  'stage4_categorical_gate_matches_archive': any(c['type']=='scientific_decision' for c in comparison['checks']) and all(not c.get('mismatched_decisions') for c in comparison['checks'] if c['type']=='scientific_decision'),
                  'stage4_gate_ci95_low': gate['ci95_low'], 'stage4_gate_ci95_high': gate['ci95_high'],
                  'strict_numerical_comparison_pass': comparison['all_comparisons_pass'],
                  'strict_numerical_failed_checks': sum(not c['pass'] for c in comparison['checks']),
                  'rtol': comparison['rtol'], 'atol': comparison['atol'],
                  'fair_neural_independent_dgp_seeds': 12, 'fair_neural_regimes': 2,
                  'fair_neural_validation': fair_validation['status'],
                  'selected_stage_retraining': selected_validations,
                  'individual_predictions_or_patient_data_included': False,
                  'scientific_note': 'Technical completion and scientific stability are reported separately. Post-review neural results are not prespecified.'}
    files['VALIDATION_REPORT.json'] = json.dumps(validation, indent=2).encode()
    decision_rows = [{'stage':'stage4','new_decision':gate['primary_gate'],
                      'archived_decision':archived_gate['primary_gate'],
                      'grid_complete':gate['grid_complete'],
                      'categorical_match':validation['stage4_categorical_gate_matches_archive'],
                      'strict_numeric_match':comparison['all_comparisons_pass'],
                      'failed_checks':validation['strict_numerical_failed_checks']}]
    for selected_stage, info in selected_validations.items():
        decision_rows.append({'stage':selected_stage,
            'new_decision':info.get('decision_status','PENDING'),
            'archived_decision':info.get('archived_decision_status',''),
            'grid_complete':info.get('grid_complete',False),
            'categorical_match':info.get('categorical_decision_matches_archive',''),
            'strict_numeric_match':info.get('strict_numerical_comparison_pass',''),
            'failed_checks':info.get('strict_numerical_failed_checks','')})
    files['STAGE_DECISION_SUMMARY.csv'] = csv_bytes(decision_rows)
    numeric_summary = []
    for stage_name, report in comparison_reports.items():
        for check in report['checks']:
            differences = check.get('numeric_differences', [])
            numeric_summary.append({'stage':stage_name,'check_type':check['type'],
                'source_file':check.get('file',''),'pass':check['pass'],
                'numeric_entries':len(differences),
                'numeric_failures':sum(not item.get('pass',True) for item in differences),
                'max_absolute_difference':max((item.get('absolute_difference',0) for item in differences),default=0),
                'categorical_mismatch_count':len(check.get('mismatched_decisions',[])),
                'reported_mismatch_count':len(check.get('mismatches',[]))})
    files['NUMERIC_DIFFERENCE_SUMMARY.csv'] = csv_bytes(numeric_summary)
    run_status = {'updated_utc':validation['created_utc'],
                  'stage4_cpu':'COMPLETE' if a.backend=='cpu' else 'SEPARATE_GPU_PACKAGE',
                  'selected_stages':selected_validations,
                  'fair_neural':'COMPLETE'}
    files['RUN_STATUS.json'] = json.dumps(run_status,indent=2).encode()
    for status_file in BASE.glob('RUN_STATUS*.json'):
        files['RUN_STATUS_HISTORY/'+status_file.name] = status_file.read_bytes()
    experiment_manifest = [
        {'experiment':'independent_frozen_stage4','source':f'kaggle:prinprin/{stage_slug}',
         'data_sha256':kernel_manifest['data_sha256'],'code_sha256':kernel_manifest['release_sha256'],
         'independent_dgp_seeds':5,'nested_network_seeds':3,'run_utc':kernel_manifest['utc']},
        {'experiment':'post_review_fair_neural','source':'7 private Kaggle kernels listed in FAIR_NEURAL_VALIDATION.json',
         'data_sha256':kernel_manifest['data_sha256'],'code_sha256':sha((BASE/'fair_neural_template.py').read_bytes()),
         'independent_dgp_seeds':12,'nested_network_seeds':1,'run_utc':fair_validation['utc']}]
    for selected_stage, info in selected_validations.items():
        experiment_manifest.append({'experiment':'independent_frozen_'+selected_stage,
            'source':f'kaggle:prinprin/b01-p1-independent-{selected_stage}-aggregate{suffix}' if info['status']=='COMPLETE' else 'PENDING',
            'data_sha256':kernel_manifest['data_sha256'],'code_sha256':kernel_manifest['release_sha256'],
            'independent_dgp_seeds':5 if selected_stage!='stage7b' else 6,
            'nested_network_seeds':2 if selected_stage in ('stage6a','stage6c','stage7b') else 3,
            'run_utc':info.get('run_utc','')})
    files['EXPERIMENT_MANIFEST.csv'] = csv_bytes(experiment_manifest)
    manifest = [{'path':name,'sha256':sha(payload),'bytes':len(payload)} for name,payload in sorted(files.items())]
    files['MANIFEST_SHA256.json'] = json.dumps({'created_utc':validation['created_utc'],'files':manifest}, indent=2).encode()
    out = BASE/('B01_P1_ACTUAL_KAGGLE_RESULTS_'+a.backend.upper()+'.zip')
    temporary = out.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=8) as z:
        for name, payload in sorted(files.items()):
            z.writestr(name, payload)
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:
            raise SystemExit('Final ZIP CRC failure')
    temporary.replace(out)
    print(out, 'entries', len(files), 'bytes', out.stat().st_size)


if __name__ == '__main__':
    main()
