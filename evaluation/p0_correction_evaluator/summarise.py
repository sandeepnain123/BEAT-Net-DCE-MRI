"""Aggregate only complete, frozen-protocol runs. Run locally after collection."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs'
cfg=json.loads((ROOT/'config.json').read_text());ids=json.loads((ROOT/'official_test_ids.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
lock=json.loads((ROOT/'freeze.json').read_text())
for name,digest in lock['files'].items():assert sha(ROOT/name)==digest,name+' changed'
rng=np.random.default_rng(cfg['statistics']['seed'])
def ci(a):
    n=len(a)
    if not n:return [None,None]
    means=[]
    for _ in range(cfg['statistics']['bootstrap_iterations']//1000):
        means.extend(a[rng.integers(0,n,size=(1000,n))].mean(1))
    return [float(x) for x in np.percentile(means,[2.5,97.5])]
data={};aggregate=[]
for target in cfg['targets']:
    name=target['name'];done=json.loads((OUT/(name+'_done.json')).read_text());path=OUT/(name+'_per_case.csv')
    assert done['status']=='DONE' and done['freeze_sha256']==sha(ROOT/'freeze.json')
    assert done['per_case_csv_sha256']==sha(path) and done['checkpoint_sha256']==lock['checkpoints'][name]['sha256']
    df=pd.read_csv(path);assert df.pid.tolist()==ids and len(df)==306
    data[name]=df.set_index('pid')
    for metric in ['dice','iou','hd95','assd']:
        a=df[metric].to_numpy();valid=a[np.isfinite(a)]
        aggregate.append(dict(model=name,metric=metric,n_total=306,n_defined=len(valid),n_undefined=306-len(valid),
          mean=float(np.mean(valid)) if len(valid) else None,sd=float(np.std(valid,ddof=1)) if len(valid)>1 else None,
          median=float(np.median(valid)) if len(valid) else None,q25=float(np.percentile(valid,25)) if len(valid) else None,
          q75=float(np.percentile(valid,75)) if len(valid) else None,mean_ci95=ci(valid)))
paired=[]
for seed in [0,1]:
    a=data[f'random_init_seed{seed}'];b=data[f'reference_seed{seed}'];assert a.index.tolist()==b.index.tolist()
    for metric in ['dice','iou','hd95','assd']:
        av=a[metric].to_numpy();bv=b[metric].to_numpy();ok=np.isfinite(av)&np.isfinite(bv);delta=av[ok]-bv[ok]
        p=float(wilcoxon(delta,alternative='two-sided',zero_method='wilcox',method='auto').pvalue) if len(delta) and np.any(delta) else (1. if len(delta) else None)
        paired.append(dict(seed=seed,metric=metric,contrast='random_init minus reference',n_pairs=len(delta),
          n_excluded=306-len(delta),mean_difference=float(np.mean(delta)) if len(delta) else None,
          paired_bootstrap_ci95=ci(delta),wilcoxon_p=p,
          interpretation='Exploratory protocol-matched comparison; historical training membership UNKNOWN and reference stage-1 lineage unresolved'))
order=sorted((i for i,x in enumerate(paired) if x['wilcoxon_p'] is not None),key=lambda i:paired[i]['wilcoxon_p'])
last=0.
for rank,i in enumerate(order):
    last=max(last,min(1.,paired[i]['wilcoxon_p']*(8-rank)));paired[i]['holm_p_family8']=last
for x in paired:x.setdefault('holm_p_family8',None)
for name,rows in [('aggregate_statistics',aggregate),('paired_comparisons',paired)]:
    (OUT/(name+'.json')).write_text(json.dumps(rows,indent=2,allow_nan=False))
    with (OUT/(name+'.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
report=dict(status='COMPLETE',historical_training_membership='UNKNOWN',leakage_free_claim=False,
 architecture_superiority_claim=False,freeze_sha256=sha(ROOT/'freeze.json'),
 no_test_driven_tuning=True,threshold=cfg['inference']['decision'],aggregate=aggregate,paired=paired)
(OUT/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps(dict(status='COMPLETE',means={x['model']+':'+x['metric']:x['mean'] for x in aggregate if x['metric']=='dice'}),indent=2))
