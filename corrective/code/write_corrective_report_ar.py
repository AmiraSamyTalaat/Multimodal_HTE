from __future__ import annotations
import json
import pandas as pd
from corrective_common import COR,now

def rd(p):return pd.read_csv(p) if p.exists() else pd.DataFrame()
def mt(df,cols):
    if df.empty:return 'لم تكتمل البيانات بعد.'
    z=df[cols].copy()
    for c in z.select_dtypes('number'):
        if c.startswith('n_'):continue
        z[c]=z[c].map(lambda x:f'{x:.4f}' if pd.notna(x) else '')
    return z.to_markdown(index=False)
cl=rd(COR/'results/clinical_adjustment/summary_results.csv')
nt=rd(COR/'results/nonoracle_tuning/summary_results.csv')
fx=rd(COR/'results/fixed_dgp_splits/summary_results.csv')
fr=rd(COR/'results/fair_neural/summary_results.csv')
pair=rd(COR/'results/fair_neural/paired_dgp_comparisons.csv')
drift=rd(COR/'reporting/old_replication_split_truth_drift.csv')
status=json.loads((COR/'RUN_STATUS.json').read_text()) if (COR/'RUN_STATUS.json').exists() else {}
matrix=rd(COR/'reviewer_objection_matrix.csv')
kcode=rd(COR/'reporting/kaggle_dataset_code_verification.csv')
cl=cl[cl.method.isin(['archived_clinical_X','archived_clinical_T','archived_full_X','archived_full_T',
    'archived_zero_effect','clinical_R_full_fitted_Rloss_selected','clinical_R_clinical_fitted_Rloss_selected'])]
nt=nt[nt.method.isin(['clinical_X_learner_NONORACLE','clinical_X_learner_ORACLE_DIAGNOSTIC',
    'full_X_learner_NONORACLE','full_X_learner_ORACLE_DIAGNOSTIC'])]
if not fx.empty:fx=fx[fx.method.isin(['zero_effect','clinical_X_learner','full_X_learner','clinical_R_full_nuisance_Rloss_selected'])]
if not fr.empty:fr=fr[fr.method.isin(['CM_OSR','uniform_same_experts','bagged_early_fusion_5fold_capacity_matched',
    'clinical_X_learner','full_X_learner','zero_effect'])]
head=f'''# مراجعة B01 benchmark التصحيحية بعد التحكيم

تاريخ الإعداد: {now()}. الحالة العامة: **{status.get('status','PARTIAL')}**. هذه تحليلات صُممت بعد التحكيم ورؤية النتائج المؤرشفة، وليست تجارب Prespecified. لم تُعدّل نتائج the benchmark الأصلية، ولم يُستخدم Paper 2 أو تُعدّل المخطوطة.

مصفوفة `reviewer_objection_matrix.csv` تربط **{len(matrix)} بندًا** من التدقيق الجنائي بالدليل، والتعديل الكتابي، أو التجربة الإضافية اللازمة. الجداول التالية تعرض المقاييس حسب بذرة DGP؛ تكرارات الشبكات وتقسيمات المرضى داخل البذرة ليست وحدات مستقلة.

## 1. الضبط السببي عند الاقتصار على clinical

العلاج والنتيجة المحاكيان يعتمدان على transcriptomic وRPPA، ولذلك Clinical X/T الأصليان لا يستخدمان كل متغيرات الإرباك. نُفّذ R-learner يقدّر propensity ونتائج الذراعين بجميع المتغيرات مع cross-fitting على TRAIN، ثم يحصر مرحلة تعلّم الأثر في clinical. اختيار Ridge تم بـR-loss من T/Y على DEV؛ استُخدمت nuisance الحقيقية في تشخيص محاكاة منفصل. حين تتغير propensity، يمثل هذا التقدير إسقاطًا سريريًا موزونًا بتباين العلاج، وليس بالضرورة E[CATE|clinical] غير الموزون.

متوسط Root-PEHE على بذور DGP الأصلية الخمس:

{mt(cl,['method','mean_root_pehe','mcse_dgp','n_dgp'])}

الفارق بين R السريري مع nuisance كاملة وZero-effect هو -0.0013 فقط (فترة ثقة مقترنة 95% من -0.0186 إلى 0.0161)؛ فلا دليل متين على تفوقه. كما أن نسخه ذات nuisance سريرية فقط حققت 0.1598 مقابل 0.1611 للنموذج ذي nuisance الكاملة. هذه نتيجة سلبية محفوظة. **المطلوب كتابيًا:** عدم وصف Clinical X/T كاختبار لضبط كل الإرباك، وعدم المبالغة في نجاح nuisance عالية الأبعاد.

## 2. حساسية تقسيم المرضى مع DGP ثابت

في التكرار السابق، تغيّر preprocessing و`reference_idx=tr` بتغير التقسيم، فتغير CATE نفسه. أعدنا حساب الحقيقة فقط عبر 30 بذرة × 3 آليات × 3 أزواج تقسيمات، على المرضى الـ736 أنفسهم، دون إعادة تدريب نماذج. متوسط RMSE لانجراف CATE:

{mt(drift.groupby('effect_regime',as_index=False).agg(mean_cate_rmse=('full_cohort_cate_rmse','mean'),fraction_changed=('full_cohort_fraction_changed_gt_1e4','mean')) if not drift.empty else drift,['effect_regime','mean_cate_rmse','fraction_changed'])}

التجربة الجديدة تولّد t وY(0) وY(1) وCATE مرة واحدة لكل بذرة وآلية، ثم تثبّتها عبر 3 تقسيمات TRAIN/DEV مع TEST نفسه. تضبط preprocessing الخاصة بالنموذج على TRAIN في كل تقسيم. الاستدلال يعتمد على 12 بذرة DGP لكل آلية؛ تقسيمات المرضى الثلاثة قياسات مرتبطة داخلها. حالة الوحدات: {status.get('experiments',{}).get('fixed_dgp_splits',{}).get('completed_units',0)}/72.

{mt(fx,['effect_regime','method','mean_root_pehe','mean_within_dgp_split_sd','mcse_dgp','n_dgp'])}

**المطلوب كتابيًا:** وصف التكرار القديم كتغير مركب في التقسيم وآلية توليد الأثر، واستخدام نتائج Fixed-DGP عند الحديث عن أثر التقسيم وحده.

## 3. ضبط معلمات قابل للتطبيق

جربنا X/T على full وclinical بثلاث قيم `min_samples_leaf` هي 4 و8 و16، مع 300 شجرة ثابتة. اختيرت المعلمات عمليًا بـR-loss المرصود على DEV مع nuisance مدربة على TRAIN؛ اختيار true DEV CATE يظهر فقط باسم `ORACLE_DIAGNOSTIC`. لم يدخل TEST في الاختيار. الميزانية 12 نموذجًا مرشحًا لكل بذرة، وليست بحثًا شاملاً.

{mt(nt,['method','mean_root_pehe','mcse_dgp','n_dgp'])}

**المطلوب كتابيًا:** الفصل بين الضبط القابل للتطبيق وبين تشخيص المحاكاة بمعرفة CATE الحقيقي، وعدم تعميم هذه الحساسية المحدودة على كل hyperparameters.

## 4. المقارنة العصبية والإنصاف الحسابي

على 12 بذرة DGP لكل من آليتين، وبذرة شبكة واحدة وتقسيم واحد، اشتركت CM-OSR وUniform في الخبراء أنفسهم، واستخدمت Bagged Early Fusion خمسة نماذج ومخرجات nuisance نفسها. طُبّق حد 250 epoch وصبر 30 وفق train-MSE على الخبراء وBagged، مع تسجيل epochs وخطوات optimizer والوقت والمعلمات. تطابقت معلمات النشر تقريبًا، لكن CM-OSR درّب 24 خبيرًا لكل تشغيل وأضاف 400 خطوة router، مقابل 5 تدريبات لـBagged. لذلك لا تكفي هذه المقارنة لنسبة أي فرق إلى Routing وحده. أُبلغت التنبؤات الخام منفصلة عن تصغير خرج مشترك اختير بـDEV R-loss؛ اختيار معامل صفر نتيجة سلبية محفوظة.

{mt(fr,['effect_regime','method','calibration','mean','mcse_dgp','n_independent_dgp'])}

الفروق المقترنة الخام، حيث القيمة السالبة تفضل CM-OSR:

{mt(pair,['effect_regime','method_a','method_b','mean','ci95_lower','ci95_upper','n_independent_dgp'])}

تنبؤات الأثر الخام خارج المجال [-1,1] بلغت في المتوسط نحو 13.5% لـCM-OSR، و50.7% لـUniform، و69.4% لـBagged. خفض clipping الأخطاء لكنه لم يجعل هذه النماذج متفوقة على خطوط الصفر/المتعلمين الكلاسيكيين. لذلك يظهر دور المعايرة والانتظام بوضوح، ولا يجوز تقديم فارق الخام على أنه دليل Routing منفرد.

**تجربة إضافية فعلًا إذا أُريد ادعاء ميزة Routing عامة:** ميزانية خطوات تدريب متساوية، وبذور شبكة متعددة، وربما مجموعة مرضى مستقلة. نطاق هذا التشغيل لا يغطي كل آليات التكرار القديم ولا يثبت التعميم السريري.

## 5. تصحيحات الإبلاغ

اجتازت إعادة الحساب سبعة فحوص: فترات الثقة القديمة استخدمت 30 متوسط بذرة DGP لكل آلية بعد جمع تقسيماتها، لا 270 وحدة مستقلة؛ وأكد فحص انجراف الحقيقة تغير CATE عبر التقسيمات. Oracle constant هو متوسط CATE الحقيقي على TEST وأفضل **ثابت** فقط تحت خطأ المربعات؛ Root-PEHE له يساوي SD(CATE الحقيقي)، ولا يمثل حدًا أدنى عامًا. ATE المقدرة هي متوسط DR pseudo-outcome عبر 5 طيات TRAIN، منفصلة عن فرق الذراعين الخام. policy regret يساوي `mean(max(tau,0))-mean(tau*1[pred>0])`، وهو مقياس محاكاة بمعرفة الحقيقة وليس منفعة سريرية مرصودة. في Stage5A severe overlap تغيّر CATE كذلك: متوسط RMSE على TEST يساوي 0.0872، وتغيّر نحو 99.3% من قيمه؛ لا يصح عزو فرق الأداء إلى overlap وحده. يتوفر اضطراب propensity معزول في النتائج السابقة.

## 6. إعادة الإنتاج والنشر

الـZIP المرشح للنشر يضم الكود، البيئة، الجداول المجمعة، الرسوم، الـmanifests وتقارير التحقق فقط. يمنع فحص الحزمة إدراج NPZ أو بيانات المرضى أو التنبؤات الفردية. حساب Kaggle `prinprin` استُخدم بخصوصية، واستُرجعت مخرجات كل دفعة المكتملة مع ZIP وبصمته. طابقت ملفات الكود الأساسية التي استُرجعت من Dataset الخاص نسخة المصدر المحلي بالبصمة SHA-256. لا توجد عملية نشر إلى Zenodo. بيانات GEO عامة، لكن [إخلاء مسؤولية NCBI](https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html) لا يحسم جميع حقوق إعادة توزيع البيانات المعالجة؛ يلزم فحص بشري قبل أي رفع عام.

**ما بقي تجربة إضافية حقًا:** تحليل تباين هرمي يشمل بذور الشبكة ومجموعات مرضى مستقلة، مطابقة ميزانية التدريب الإجمالية قبل ادعاء أثر Routing، حساسية عصبية أوسع، واختبار إعادة تشغيل مستقل للقرارات المجمدة إذا أُريد ادعاء التطابق الدقيق. باقي البنود الموثقة في المصفوفة تحتاج تصحيحًا كتابيًا وحدود تفسير واضحة، لا حذف نتائج سلبية.
'''
(COR/'CORRECTIVE_REVIEW_REPORT_AR.md').write_text(head,encoding='utf-8')
print(COR/'CORRECTIVE_REVIEW_REPORT_AR.md')
