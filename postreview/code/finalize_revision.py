from __future__ import annotations
import hashlib, json, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

REV=Path(__file__).resolve().parents[1]
NOW=lambda:datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def status(stage):
    p=REV/'results'/stage/'RUN_STATUS.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {'status':'MISSING'}
def results():
    frames=[]
    mappings=[('baselines','per_seed_results.csv','root_pehe','policy_regret'),
              ('oracle_fairness','per_seed_results.csv','test_root_pehe','test_policy_regret'),
              ('fairness','per_seed_results.csv','root_pehe','policy_regret'),
              ('replication','per_seed_results.csv','root_pehe','policy_regret')]
    for stage,filename,metric,policy in mappings:
        p=REV/'results'/stage/filename
        if not p.exists():continue
        df=pd.read_csv(p)
        if metric not in df:continue
        out=pd.DataFrame({'stage':stage,'dgp_seed':df.get('dgp_seed'),
            'network_seed':df.get('net_seed',pd.Series([pd.NA]*len(df))),
            'patient_split_seed':df.get('patient_split_seed',pd.Series([20260918]*len(df))),
            'effect_regime':df.get('effect_regime',pd.Series(['PRIMARY']*len(df))),
            'method':df['method'],'root_pehe':df[metric],
            'policy_regret':df.get(policy,pd.Series([np.nan]*len(df))),
            'source_table':str(p.relative_to(REV)).replace('\\','/')})
        frames.append(out)
    p=REV/'results/missingness/per_seed_results.csv'
    if p.exists():
        df=pd.read_csv(p)
        frames.append(pd.DataFrame({'stage':'missingness','dgp_seed':df.dgp_seed,
          'network_seed':pd.Series([pd.NA]*len(df)),
          'patient_split_seed':pd.Series([20260918]*len(df)),
          'effect_regime':df.condition,'method':df.method,
          'root_pehe':df.root_pehe,'policy_regret':df.policy_regret,
          'source_table':str(p.relative_to(REV)).replace('\\','/')}))
        cluster=df.groupby(['condition','method','dgp_seed'],as_index=False).agg(
          root_pehe=('root_pehe','mean'),policy_regret=('policy_regret','mean'))
        cluster.to_csv(REV/'results/missingness/dgp_cluster_means.csv',index=False)
        sm=cluster.groupby(['condition','method'],as_index=False).agg(
          mean_root_pehe=('root_pehe','mean'),sd_across_dgp=('root_pehe','std'),
          mean_policy_regret=('policy_regret','mean'),n_independent_dgp=('dgp_seed','nunique'))
        sm['mcse_root_pehe']=sm.sd_across_dgp/np.sqrt(sm.n_independent_dgp)
        sm.to_csv(REV/'results/missingness/summary_results.csv',index=False)
    p=REV/'results/classical_sensitivity/per_seed_results.csv'
    if p.exists():
        df=pd.read_csv(p)
        frames.append(pd.DataFrame({'stage':'classical_sensitivity','dgp_seed':df.dgp_seed,
          'network_seed':pd.Series([pd.NA]*len(df)),
          'patient_split_seed':pd.Series([20260918]*len(df)),
          'effect_regime':'PRIMARY','method':df.feature_set+'_'+df.learner+'_DEV_selected',
          'root_pehe':df.test_root_pehe,'policy_regret':df.test_policy_regret,
          'source_table':str(p.relative_to(REV)).replace('\\','/')}))
        sm=df.groupby(['feature_set','learner'],as_index=False).agg(
          mean_test_root_pehe=('test_root_pehe','mean'),sd_across_dgp=('test_root_pehe','std'),
          mean_dev_root_pehe=('selected_dev_root_pehe','mean'),
          mean_policy_regret=('test_policy_regret','mean'),
          n_independent_dgp=('dgp_seed','nunique'))
        sm['mcse_test_root_pehe']=sm.sd_across_dgp/np.sqrt(sm.n_independent_dgp)
        sm.to_csv(REV/'results/classical_sensitivity/summary_results.csv',index=False)
    p=REV/'results/ridge_extension/per_run_results.csv'
    if p.exists():
        df=pd.read_csv(p)
        df['method']='DR_Ridge_'+df.nuisance+'_extended_DEV_selected'
        per=df.groupby(['dgp_seed','method'],as_index=False).agg(
          root_pehe=('test_root_pehe','mean'),policy_regret=('test_policy_regret','mean'),
          network_repeats=('net_seed','nunique'))
        per.to_csv(REV/'results/ridge_extension/per_seed_results.csv',index=False)
        frames.append(pd.DataFrame({'stage':'ridge_extension','dgp_seed':per.dgp_seed,
          'network_seed':pd.Series([pd.NA]*len(per)),
          'patient_split_seed':pd.Series([20260918]*len(per)),
          'effect_regime':'PRIMARY','method':per.method,
          'root_pehe':per.root_pehe,'policy_regret':per.policy_regret,
          'source_table':str(p.relative_to(REV)).replace('\\','/')}))
        sm=per.groupby('method',as_index=False).agg(
          mean_test_root_pehe=('root_pehe','mean'),sd_across_dgp=('root_pehe','std'),
          mean_policy_regret=('policy_regret','mean'),n_independent_dgp=('dgp_seed','nunique'))
        sm['mcse_test_root_pehe']=sm.sd_across_dgp/np.sqrt(sm.n_independent_dgp)
        sm.to_csv(REV/'results/ridge_extension/summary_results.csv',index=False)
    if frames:
        allrows=pd.concat(frames,ignore_index=True)
        stage_code={'baselines':'run_baselines.py','oracle_fairness':'run_oracle_fairness.py',
          'fairness':'kaggle_bagged_fairness_template.py',
          'missingness':'kaggle_missingness.py',
          'classical_sensitivity':'kaggle_classical_sensitivity.py',
          'ridge_extension':'kaggle_ridge_extension.py',
          'replication':'run_replication.py'}
        allrows['code_sha256']=allrows.stage.map(lambda k:sha(REV/'postreview_code'/stage_code[k]))
        allrows['run_utc']=allrows.stage.map(lambda k:status(k).get('finished_utc',
                                        status(k).get('last_updated_utc','')))
        allrows['data_sha256']=next((r.original_sha256 for r in pd.read_csv(REV/'source_provenance.csv').itertuples()
            if str(r.relative_path).replace('\\','/').endswith('ispy2_core_multimodal.npz')),None)
        allrows.to_csv(REV/'per_seed_results.csv',index=False)
        summary=(allrows.groupby(['stage','effect_regime','method'],dropna=False,as_index=False)
            .agg(mean_root_pehe=('root_pehe','mean'),sd_rows=('root_pehe','std'),
                 mean_policy_regret=('policy_regret','mean'),
                 n_dgp=('dgp_seed','nunique'),n_rows=('root_pehe','count')))
        summary.to_csv(REV/'summary_results.csv',index=False)
def manifest():
    rows=[]
    stage_code={'baselines':'run_baselines.py','oracle_fairness':'run_oracle_fairness.py',
      'fairness':'kaggle_bagged_fairness_template.py',
      'additional':'run_additional_diagnostics.py','imputation':'run_imputation_diagnostics.py',
      'missingness':'kaggle_missingness.py','classical_sensitivity':'kaggle_classical_sensitivity.py',
      'ridge_extension':'kaggle_ridge_extension.py','replication':'run_replication.py'}
    data_hash=next((x['original_sha256'] for x in pd.read_csv(REV/'source_provenance.csv').to_dict('records')
                    if str(x.get('relative_path','')).replace('\\','/').endswith('ispy2_core_multimodal.npz')),None)
    for folder,stage in [('baselines','BASELINES'),('oracle_fairness','ORACLE_REGULARIZATION'),
                         ('fairness','FAIR_BAGGING'),('additional','ADDITIONAL'),
                         ('imputation','IMPUTATION'),('missingness','MISSINGNESS'),
                         ('classical_sensitivity','CLASSICAL_SENSITIVITY'),
                         ('ridge_extension','RIDGE_EXTENSION'),('replication','REPLICATION')]:
        s=status(folder)
        rows.append({'stage':stage,'unit':'aggregate','status':s.get('status'),
            'dgp_seed':'','network_seed':'','patient_split_seed':'','effect_regime':'',
            'data_sha256':data_hash,'code_sha256':sha(REV/'postreview_code'/stage_code[folder]),
            'started_utc':'','finished_utc':s.get('finished_utc',s.get('last_updated_utc','')),
            'source':'original B01 local snapshot / private prinprin kernel outputs'})
    for p in (REV/'results/baselines').glob('DGP*_manifest.json'):
        d=json.loads(p.read_text(encoding='utf-8'))
        rows.append({'stage':'BASELINES','unit':p.stem,'status':d.get('status'),
          'dgp_seed':d.get('dgp_seed'),'network_seed':'','patient_split_seed':d.get('split_seed'),
          'effect_regime':'PRIMARY','data_sha256':d.get('source_data_sha256'),
          'code_sha256':sha(REV/'postreview_code/run_baselines.py'),
          'started_utc':d.get('started_utc'),'finished_utc':d.get('finished_utc'),
          'source':'15 archived test_predictions.npz; TRAIN fitted diagnostics'})
    for p in (REV/'results/oracle_fairness/runs').glob('*/COMPLETE.json') if (REV/'results/oracle_fairness/runs').exists() else []:
        d=json.loads(p.read_text(encoding='utf-8'))
        rows.append({'stage':'ORACLE_REGULARIZATION','unit':p.parent.name,'status':d.get('status'),
          'dgp_seed':d.get('dgp_seed'),'network_seed':d.get('net_seed'),'patient_split_seed':20260918,
          'effect_regime':'PRIMARY','data_sha256':d.get('data_sha256',data_hash),
          'code_sha256':d.get('code_sha256',sha(REV/'postreview_code/run_oracle_fairness.py')),
          'started_utc':d.get('started_utc'),'finished_utc':d.get('finished_utc'),
          'source':'original processed NPZ; TRAIN/DEV nuisance diagnostic'})
    p=REV/'results/fairness/kaggle_batch_manifest.csv'
    if p.exists():
        for d in pd.read_csv(p).to_dict('records'):
            rows.append({'stage':'FAIR_BAGGING','unit':d['kernel'],'status':'COMPLETE',
              'dgp_seed':d['dgp_seed'],'network_seed':'17,29,43','patient_split_seed':20260918,
              'effect_regime':'PRIMARY','data_sha256':d['data_sha256'],
              'code_sha256':sha(REV/'postreview_code/kaggle_bagged_fairness_template.py'),
              'started_utc':d['started_utc'],'finished_utc':d['finished_utc'],
              'source':'private Kaggle kernel durable output; SHA-256 '+d['metrics_sha256']})
    for p in (REV/'results/replication/runs').glob('*/COMPLETE.json'):
        d=json.loads(p.read_text(encoding='utf-8'))
        rows.append({'stage':'REPLICATION','unit':d.get('tag',p.parent.name),'status':d.get('status'),
          'dgp_seed':d.get('dgp_seed'),'network_seed':'','patient_split_seed':d.get('patient_split_seed'),
          'effect_regime':d.get('effect_regime'),'data_sha256':d.get('data_sha256'),
          'code_sha256':d.get('code_sha256'),'started_utc':d.get('started_utc'),
          'finished_utc':d.get('finished_utc'),
          'execution_code_sha256':d.get('kaggle_execution_code_sha256',''),
          'source':'original processed NPZ; Stage7 generated effect structure'})
    for p in (REV/'results/missingness').glob('DGP*/COMPLETE.json'):
        d=json.loads(p.read_text(encoding='utf-8'))
        rows.append({'stage':'MISSINGNESS','unit':p.parent.name,'status':d.get('status'),
          'dgp_seed':d.get('dgp_seed'),'network_seed':'','patient_split_seed':20260918,
          'effect_regime':'MCAR_40,MAR_CLINICAL','data_sha256':d.get('data_sha256'),
          'code_sha256':sha(REV/'postreview_code/kaggle_missingness.py'),
          'started_utc':'','finished_utc':d.get('finished_utc'),
          'source':'private Kaggle kernel prinprin/b01-p1-postreview-missingness'})
    for p in (REV/'results/classical_sensitivity').glob('DGP*/COMPLETE.json'):
        d=json.loads(p.read_text(encoding='utf-8'))
        rows.append({'stage':'CLASSICAL_SENSITIVITY','unit':p.parent.name,'status':d.get('status'),
          'dgp_seed':d.get('dgp_seed'),'network_seed':'','patient_split_seed':20260918,
          'effect_regime':'PRIMARY','data_sha256':d.get('data_sha256'),
          'code_sha256':sha(REV/'postreview_code/kaggle_classical_sensitivity.py'),
          'started_utc':'','finished_utc':d.get('finished_utc'),
          'source':'private Kaggle kernel prinprin/b01-p1-postreview-classical-sensitivity'})
    for p in (REV/'results/ridge_extension').glob('DGP*/COMPLETE.json'):
        d=json.loads(p.read_text(encoding='utf-8'))
        rows.append({'stage':'RIDGE_EXTENSION','unit':p.parent.name,'status':d.get('status'),
          'dgp_seed':d.get('dgp_seed'),'network_seed':d.get('net_seed'),'patient_split_seed':20260918,
          'effect_regime':'PRIMARY','data_sha256':d.get('data_sha256'),
          'code_sha256':sha(REV/'postreview_code/kaggle_ridge_extension.py'),
          'started_utc':'','finished_utc':d.get('finished_utc'),
          'source':'private Kaggle kernel prinprin/b01-p1-postreview-ridge-extension'})
    overlap=REV/'results/additional/CLEAN_OVERLAP_STATUS.json'
    if overlap.exists():
        d=json.loads(overlap.read_text(encoding='utf-8'))
        rows.append({'stage':'CLEAN_OVERLAP','unit':'5 DGP × 3 splits × 2 conditions',
          'status':d.get('status'),'dgp_seed':'101,211,307,401,503','network_seed':'',
          'patient_split_seed':'20260918','effect_regime':'PRIMARY,SEVERE_PROPENSITY_ONLY',
          'data_sha256':d.get('data_sha256'),'code_sha256':d.get('code_sha256'),
          'started_utc':'','finished_utc':d.get('finished_utc'),
          'source':'local source snapshot; shared treatment uniforms; fixed CATE'})
    pd.DataFrame(rows).to_csv(REV/'experiment_manifest.csv',index=False)
def reports():
    fair=pd.read_csv(REV/'results/fairness/summary_results.csv').set_index('method')
    oracle=pd.read_csv(REV/'results/oracle_fairness/summary_results.csv').set_index('method')
    miss=status('missingness'); sensitivity=status('classical_sensitivity')
    extension=status('ridge_extension'); rep=status('replication')
    replication_note=('اكتملت جميع الوحدات الكلاسيكية؛ تُعرض فترات الثقة في تقرير التكرار.'
                      if rep.get('status')=='COMPLETE' else
                      'لا يصلح الملخص الجزئي كاستنتاج نهائي لـ30 بذرة.')
    missingness=pd.read_csv(REV/'results/missingness/summary_results.csv') if (REV/'results/missingness/summary_results.csv').exists() else pd.DataFrame()
    txt=f'''# B01 benchmark — تشخيص Oracle والانتظامية والمقارنة المنصفة بعد التحكيم

تاريخ التجميع UTC: {NOW()}. هذه تجارب Post-review. بذور DGP الخمس هي وحدات المقارنة المستقلة؛ البذور الشبكية الثلاث داخل كل DGP مكررات تدريب.

## المقارنة الرئيسية: Root-PEHE على TEST

| النموذج | المتوسط | MCSE على مستوى DGP | نسبة التنبؤات خارج [-1,1] |
|---|---:|---:|---:|
'''
    for m in ['X_learner','CM_OSR','uniform_all_experts_ablation','early_fusion_DR_capacity_matched',
              'bagged_early_fusion','single_early_stopped_early_fusion']:
        d=fair.loc[m];txt+=f"| {m} | {d.mean_root_pehe:.3f} | {d.mcse_root_pehe:.3f} | {d.mean_outside_fraction:.1%} |\n"
    txt+=f'''
دُرّب Bagged Early Fusion بخمسة نماذج على طيات TRAIN، بإجمالي نحو 127,025 معاملًا مقابل 126,440 في CM-OSR المؤرشف، وبنفس إجراء nuisance DR ذي الطيات الخمس وإيقاف مبكر مبني على خسارة التدريب (حد 250 حقبة، صبر 30). بلغ متوسط Root-PEHE له {fair.loc['bagged_early_fusion'].mean_root_pehe:.3f}، مقابل {fair.loc['early_fusion_DR_capacity_matched'].mean_root_pehe:.3f} للـEarly Fusion المؤرشف و{fair.loc['CM_OSR'].mean_root_pehe:.3f} لـCM-OSR. الضابط ذو النموذج الواحد والإيقاف المبكر: {fair.loc['single_early_stopped_early_fusion'].mean_root_pehe:.3f}. لا تثبت هذه المقارنة أن Routing هو سبب الفرق؛ وX-learner أفضل من CM-OSR في هذا الإعداد.

## nuisance والانتظامية في DR

متوسط Root-PEHE لـDR Ridge باستخدام nuisance fitted ومعامل اختير على DEV: {oracle.loc['DR_Ridge_fitted_DEV_selected'].mean_test_root_pehe:.3f}؛ وعند استخدام nuisance Oracle التشخيصية: {oracle.loc['DR_Ridge_oracle_DEV_selected'].mean_test_root_pehe:.3f}. قيم e وmu الحقيقية متاحة في المحاكاة فقط وليست نموذجًا قابلًا للتطبيق. اختارت التشغيلات الـ15 كلها alpha=100000 من الشبكة المحددة على DEV، ثم قُيّم TEST. عند alpha=10 الأصلي، بلغ متوسط fitted DR نحو 5.553. متوسط Brier لميل المعالجة fitted على DEV يساوي 0.344 مقابل 0.193 للميل الحقيقي؛ ميل المعايرة 0.121 مقابل 1.151 للميل الحقيقي وAUC للنموذج fitted يساوي 0.579، مع clipping واسع عند 0.05 و0.95. ترد توزيعات pseudo-outcome وكل قيم alpha في ملفات CSV.

يغيّر سيناريو Stage5A ذي التداخل الشديد معادلة CATE أيضًا في الكود المؤرشف؛ لذلك لا تعزل تلك المقارنة أثر التداخل. حُفظت مؤشرات التوافر، والمفقودات، وESS، وتوزيعات CATE، وTPR/FPR لبوابة المنفعة في نتائج التشخيصات الإضافية. لم تقبل بوابة DEV أي مرشح في تشغيلات Stage7B المؤرشفة.

في اختبار Post-review منفصل للتداخل وحده، بقيت بصمة CATE مطابقة تمامًا؛ ومع زيادة مقياس الميل انخفض متوسط ESS في TEST للمعالجين من 60.2 إلى 46.7 وللضبط من 56.5 إلى 32.8. لم يُجرَ تدريب نموذج جديد تحت هذا الاضطراب.

في إعادة بناء السمات وحدها، كان تعويض الجيران السريريين العشرة أضعف من الصفر بعد توحيد TRAIN. لكن على مستوى X-learner، حسّن تعويض جيران TRAIN Root-PEHE مقابل الصفر في MCAR40 وMAR. حسّنت مؤشرات التوافر النتيجة أكثر في MCAR40، وكانت أضعف قليلًا من تعويض الجيران في MAR. حُفظ كلا النوعين من النتائج. حالة تشغيل المفقودات على Kaggle: {miss.get('status','MISSING')}.

تحسّن Root-PEHE هنا لا يعني تحسّن كل المقاييس: في MCAR40 ارتفع ندم السياسة من 0.0229 مع تعويض الصفر إلى 0.0293 مع الجيران، وفي MAR من 0.0279 إلى 0.0311. تُظهر الجداول القيم لكل DGP.

حساسية `min_samples_leaf` لنماذج X/T (4 و8 و16، مع 300 شجرة ثابتة) اختيرت على DEV؛ الحالة {sensitivity.get('status','MISSING')}. تتضمن `results/classical_sensitivity/` شبكة DEV ونتيجة TEST للمرشح المختار.

بعد اختيار DEV، بلغ متوسط Root-PEHE للـX السريري 0.1571 مقابل 0.1623 لمرجع الصفر؛ تحسن في أربع من خمس بذور DGP فقط، والفارق المتوسط -0.0053 مع MCSE نحو 0.0028، فلا يُقدَّم كدليل حاسم على التفوق.

لأن alpha=100000 وقع عند حد شبكة DR الأولى، أضيفت شبكة أوسع حتى 1e8 وحد النموذج الثابت. حالة هذه التجربة: {extension.get('status','MISSING')}. لا تُفسَّر النتيجة الأولى كاختيار نهائي للانتظامية قبل اكتمالها.

حالة التكرار المستقل: {rep.get('completed_units',0)}/{rep.get('expected_units',270)} وحدة. {replication_note}
'''
    if extension.get('status')=='COMPLETE':
        ext=pd.read_csv(REV/'results/ridge_extension/summary_results.csv').set_index('method')
        fits=ext.loc['DR_Ridge_fitted_extended_DEV_selected'].mean_test_root_pehe
        ora=ext.loc['DR_Ridge_oracle_extended_DEV_selected'].mean_test_root_pehe
        txt+=f'\nفي الشبكة الموسعة، بلغ متوسط Root-PEHE للنموذج fitted {fits:.3f} ولـOracle التشخيصي {ora:.3f}. تُظهر ملفات DEV اختيار alpha لكل تشغيل، بما في ذلك حد النموذج الثابت.\n'
    (REV/'reports/oracle_fairness_postreview_AR.md').write_text(txt,encoding='utf-8')
    completion_phrase=('اكتملت بذور DGP الثلاثون للنماذج الكلاسيكية.' if rep.get('status')=='COMPLETE'
                       else 'الملخص الجزئي استكشافي إلى اكتمال بذور DGP الثلاثين.')
    reptext=f'''# التكرار المستقل بعد التحكيم

المنجز: {rep.get('completed_units',0)}/{rep.get('expected_units',270)} وحدة، الحالة {rep.get('status')}. التصميم: 30 بذرة DGP مستقلة، وثلاثة تقسيمات على مستوى المرضى، وثلاث آليات CATE (لا أثر جزيئي إضافي، أثر Transcriptome، وتآزر جزيئي). تسحب الآليتان الأخيرتان إسقاطات تأثير مرتبطة ببذرة DGP، فتتغير بنية الأثر وليس فقط سحوبات Bernoulli للعلاج. تدرب كل وحدة X/T الكاملة والسريرية وDR Ridge؛ تختار alpha على DEV ثم تقيّم TEST. ثابت Oracle المحسوب من TEST مرجع غير قابل للتطبيق.

وحدة الاستقلال في MCSE هي بذرة DGP. تقسيمات المرضى تعيد استخدام مجموعة المرضى الثابتة وتُوسّط داخل كل DGP؛ لذلك تصف فترات عدم اليقين تباين المحاكاة المشروط بهذه المجموعة، ولا تصف تعميمًا على مرضى مستقلين جدد. {completion_phrase} لم يُعد تشغيل CM-OSR العصبي على جميع وحدات التكرار؛ يرد ذلك في تقرير النواقص.
'''
    if rep.get('status')=='COMPLETE':
        ci=pd.read_csv(REV/'results/replication/uncertainty_intervals.csv')
        names=['zero_effect','clinical_X_learner','full_X_learner','DR_Ridge_DEV_selected']
        reptext+='\n## نتائج 30 DGP: Root-PEHE وفترة ثقة 95% مشروطة\n\n'
        reptext+='| آلية CATE | النموذج | المتوسط | الحد الأدنى | الحد الأعلى |\n|---|---|---:|---:|---:|\n'
        for d in ci[ci.method.isin(names)].sort_values(['effect_regime','method']).itertuples():
            reptext+=f'| {d.effect_regime} | {d.method} | {d.mean_root_pehe:.3f} | {d.ci95_lower:.3f} | {d.ci95_upper:.3f} |\n'
    (REV/'reports/replication_postreview_AR.md').write_text(reptext,encoding='utf-8')
    gaps=[]
    if rep.get('status')!='COMPLETE':gaps.append(f"التكرار المستقل {rep.get('completed_units',0)}/{rep.get('expected_units',270)} وحدة؛ استدلال 30 DGP لم يكتمل بعد.")
    if miss.get('status')!='COMPLETE':gaps.append('تجربة المفقودات والتعويض على مستوى النموذج في Kaggle معلقة أو فاشلة.')
    if sensitivity.get('status')!='COMPLETE':gaps.append('حساسية معاملات X/T الكلاسيكية معلقة أو فاشلة.')
    if extension.get('status')!='COMPLETE':gaps.append('تمديد شبكة DR Ridge لما بعد alpha=100000 وحد النموذج الثابت لم يكتمل بعد.')
    gaps.extend([
      'يقارن التكرار X/T/DR الكلاسيكية عبر ثلاث آليات أثر؛ لم يُنفذ تشغيل CM-OSR العصبي على كل بذور DGP الثلاثين وتقسيماتها وآلياتها.',
      'سيناريو Stage5A المؤرشف للتداخل الشديد يغيّر بنية CATE؛ أُجري اضطراب تشخيصي مستقل للميل فقط، لكن لم تُقارن أخطاء النماذج بعد إعادة التدريب تحت هذا الاضطراب.',
      'يستخدم Bagged Early Fusion إيقافًا مبكرًا بخسارة التدريب؛ لم تُشغّل شبكة إيقاف مبكر مختارة على DEV لكل نموذج عصبي.',
      'الاستدلال بالمحاكاة مشروط بمجموعة متغيرات مرضى I-SPY2 الثابتة؛ بذور DGP الثلاثون ليست 30 مجموعة مرضى مستقلة.',
      'لا تتضمن حزم ZIP المعدة للتسليم ملفات المرضى أو التنبؤات الفردية؛ يجب مراجعة حقوق إعادة التوزيع قبل النشر الخارجي.'
    ])
    (REV/'missing_requirements_report.md').write_text('# المتطلبات المتبقية وحدود التفسير\n\n'+'\n'.join(f'- {g}' for g in gaps)+'\n',encoding='utf-8')
def figures():
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        df=pd.read_csv(REV/'results/fairness/summary_results.csv').sort_values('mean_root_pehe')
        fig,ax=plt.subplots(figsize=(9,4.8));ax.barh(df.method,df.mean_root_pehe,xerr=1.96*df.mcse_root_pehe,color='#386b9b')
        ax.invert_yaxis();ax.set_xlabel('Mean TEST Root-PEHE ± 1.96 × DGP-cluster MCSE')
        ax.set_title('Post-review fair comparison (5 DGP seeds)')
        fig.tight_layout();fig.savefig(REV/'reports/fairness_root_pehe.png',dpi=170);plt.close(fig)
        grid=pd.read_csv(REV/'results/oracle_fairness/ridge_sensitivity.csv')
        fig,ax=plt.subplots(figsize=(7,4.5))
        for nuisance,color in [('fitted','#a13f42'),('oracle','#286a6a')]:
            g=grid[grid.nuisance==nuisance].groupby('alpha',as_index=False).agg(
              dev_root_pehe=('dev_root_pehe','mean'))
            ax.plot(g.alpha,g.dev_root_pehe,marker='o',label=nuisance,color=color)
        ax.set_xscale('log');ax.set_xlabel('Ridge alpha (selected on DEV)')
        ax.set_ylabel('Mean DEV Root-PEHE');ax.set_title('DR Ridge regularization sensitivity')
        ax.legend();fig.tight_layout();fig.savefig(REV/'reports/ridge_dev_sensitivity.png',dpi=170);plt.close(fig)
        p=REV/'results/replication/uncertainty_intervals.csv'
        if p.exists():
            rep=pd.read_csv(p)
            rep=rep[rep.inference_ready & rep.method.isin(['clinical_X_learner','full_X_learner','DR_Ridge_DEV_selected'])]
            if len(rep)==9:
                regimes=['NO_INCREMENTAL_MOLECULAR_SIGNAL','TRANSCRIPTOME_INCREMENTAL','JOINT_MOLECULAR_SYNERGY']
                methods=['clinical_X_learner','full_X_learner','DR_Ridge_DEV_selected']
                fig,ax=plt.subplots(figsize=(9,4.5));x=np.arange(3)
                for i,m in enumerate(methods):
                    d=rep.set_index(['effect_regime','method'])
                    vals=[d.loc[(r,m)] for r in regimes]
                    means=[v.mean_root_pehe for v in vals]
                    errors=[v.t_critical_95*v.mcse for v in vals]
                    ax.bar(x+(i-1)*.25,means,width=.24,yerr=errors,label=m)
                ax.set_xticks(x,regimes,rotation=12,ha='right')
                ax.set_ylabel('Mean TEST Root-PEHE ± 95% DGP-cluster CI')
                ax.set_title('30-DGP replication across CATE mechanisms')
                ax.legend();fig.tight_layout();fig.savefig(REV/'reports/replication_root_pehe.png',dpi=170);plt.close(fig)
    except ImportError:pass
def replication_uncertainty():
    p=REV/'results/replication/dgp_cluster_means.csv'
    if not p.exists():return
    from scipy.stats import t
    df=pd.read_csv(p)
    sm=df.groupby(['effect_regime','method'],as_index=False).agg(
      mean_root_pehe=('root_pehe','mean'),sd_across_dgp=('root_pehe','std'),
      n_independent_dgp=('dgp_seed','nunique'))
    sm['mcse']=sm.sd_across_dgp/np.sqrt(sm.n_independent_dgp)
    sm['t_critical_95']=sm.n_independent_dgp.map(lambda n:t.ppf(.975,n-1) if n>1 else np.nan)
    sm['ci95_lower']=sm.mean_root_pehe-sm.t_critical_95*sm.mcse
    sm['ci95_upper']=sm.mean_root_pehe+sm.t_critical_95*sm.mcse
    sm['inference_ready']=sm.n_independent_dgp.eq(30) & (status('replication').get('status')=='COMPLETE')
    sm['uncertainty_unit']='DGP seed; patient splits averaged within DGP; fixed cohort'
    sm.to_csv(REV/'results/replication/uncertainty_intervals.csv',index=False)
    paired=[]
    wide=df.pivot(index=['effect_regime','dgp_seed'],columns='method',values='root_pehe')
    for regime in df.effect_regime.unique():
        block=wide.loc[regime]
        for candidate,comparator in [('clinical_X_learner','full_X_learner'),
                                     ('clinical_X_learner','zero_effect'),
                                     ('DR_Ridge_DEV_selected','clinical_X_learner')]:
            d=(block[candidate]-block[comparator]).dropna()
            se=float(d.std(ddof=1)/np.sqrt(len(d)))
            critical=float(t.ppf(.975,len(d)-1))
            paired.append({'effect_regime':regime,'candidate':candidate,'comparator':comparator,
              'mean_paired_root_pehe_difference':float(d.mean()),'mcse_dgp_cluster':se,
              'ci95_lower':float(d.mean()-critical*se),'ci95_upper':float(d.mean()+critical*se),
              'n_independent_dgp':len(d),'inference_ready':len(d)==30 and status('replication').get('status')=='COMPLETE'})
    pd.DataFrame(paired).to_csv(REV/'results/replication/paired_dgp_comparisons.csv',index=False)
def package(name,groups):
    dest=REV/'packages'/name;dest.parent.mkdir(exist_ok=True)
    selected=set()
    common=['PROVENANCE.json','E_MIRROR_PROVENANCE.json','source_provenance.csv','RUN_STATUS.json','per_seed_results.csv',
            'summary_results.csv','experiment_manifest.csv','missing_requirements_report.md',
            'README_RESUME_AR.md','requirements_postreview.txt','RUN_RESUME_PAPER1.cmd']
    for rel in common:
        p=REV/rel
        if p.exists():selected.add(p)
    for pattern in groups:
        for p in REV.glob(pattern):
            if p.is_file():selected.add(p)
    safe=[]
    for p in selected:
        rel=p.relative_to(REV).as_posix()
        if p.suffix.lower() in {'.npz','.npy','.pkl','.joblib','.zip'}:continue
        if 'utility_gate_patient_audit' in rel or '/02_DATA/' in rel:continue
        if '/predictions/' in rel or '/source_snapshot/04_RUNS/' in rel:continue
        safe.append(p)
    tmp=dest.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(safe):z.write(p,p.relative_to(REV).as_posix())
    with zipfile.ZipFile(tmp) as z:
        if z.testzip() is not None:raise RuntimeError('ZIP integrity failure '+str(dest))
    tmp.replace(dest)
    return {'file':dest.name,'sha256':sha(dest),'bytes':dest.stat().st_size,'entries':len(safe)}
def main():
    results();manifest();replication_uncertainty();reports();figures()
    stages={k:status(k) for k in ['baselines','oracle_fairness','fairness','additional','imputation','missingness','classical_sensitivity','ridge_extension','replication']}
    clean=REV/'results/additional/CLEAN_OVERLAP_STATUS.json'
    stages['clean_overlap']=json.loads(clean.read_text(encoding='utf-8')) if clean.exists() else {'status':'MISSING'}
    overall='COMPLETE_WITH_DECLARED_GAPS' if all(x.get('status')=='COMPLETE' for x in stages.values()) else 'PARTIAL'
    write(REV/'RUN_STATUS.json',{'status':overall,'updated_utc':NOW(),'stages':stages,
       'scientific_scope':'Post-review; original prespecified outputs untouched',
       'independence':'DGP clusters, not network seeds; replication MCSE conditional on fixed patient cohort'})
    bundles={}
    bundles['baselines']=package('B01_P1_BASELINES_POSTREVIEW.zip',[
      'postreview_code/run_baselines.py','source_snapshot/00_CONFIG/**/*',
      'source_snapshot/01_CODE/**/*','results/baselines/*','reports/baselines_postreview_AR.md'])
    bundles['oracle_fairness']=package('B01_P1_ORACLE_FAIRNESS_POSTREVIEW.zip',[
      'postreview_code/*.py','kaggle_bundle/**/*.py','kaggle_bundle/**/kernel-metadata.json',
      'source_snapshot/00_CONFIG/**/*','source_snapshot/01_CODE/**/*',
      'results/oracle_fairness/**/*.csv','results/oracle_fairness/**/*.json',
      'results/fairness/*','results/additional/*','results/imputation/*','results/missingness/*',
      'results/classical_sensitivity/**/*',
      'results/ridge_extension/**/*',
      'runs/kaggle_fairness_dgp*/postreview_results/**/COMPLETE.json',
      'runs/kaggle_fairness_dgp*/postreview_results/**/*training_history.csv',
      'runs/kaggle_fairness_dgp*/postreview_results/RUN_STATUS.json',
      'reports/oracle_fairness_postreview_AR.md','reports/fairness_root_pehe.png',
      'reports/ridge_dev_sensitivity.png',
      'runs/kaggle_*/*.log','runs/kaggle_*/failed_attempts/**/*.log',
      'logs/*.jsonl','logs/*.log'])
    bundles['replication']=package('B01_P1_REPLICATION_POSTREVIEW.zip',[
      'postreview_code/run_replication.py','source_snapshot/01_CODE/cm_osr/stage7_dgp.py',
      'results/replication/**/*.csv','results/replication/**/*.json',
      'results/replication/**/ERROR.txt','reports/replication_postreview_AR.md',
      'reports/replication_root_pehe.png','runs/kaggle_replication_*/*.log',
      'runs/kaggle_replication_*/MERGED.json','logs/harvest_replication*.log'])
    bundles['all']=package('B01_P1_REVISION_RESULTS_ALL.zip',[
      'postreview_code/*.py','kaggle_bundle/**/*.py','kaggle_bundle/**/kernel-metadata.json',
      'source_snapshot/00_CONFIG/**/*','source_snapshot/01_CODE/**/*',
      'results/**/*.csv','results/**/*.json','results/**/ERROR.txt',
      'reports/*.md','reports/*.png','logs/*.jsonl','logs/*.log',
      'runs/kaggle_*/*.log','runs/kaggle_*/failed_attempts/**/*.log',
      'runs/kaggle_fairness_dgp*/postreview_results/**/COMPLETE.json'])
    write(REV/'packages/PACKAGE_MANIFEST.json',bundles)
    print(json.dumps({'status':overall,'replication_completed':stages['replication'].get('completed_units'),
                      'bundles':bundles},ensure_ascii=False))
if __name__=='__main__':main()
