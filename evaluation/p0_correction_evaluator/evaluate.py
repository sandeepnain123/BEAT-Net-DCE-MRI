"""Frozen correction-only evaluator. No training, model selection, or test tuning."""
import os,socket,sys,json,csv,hashlib,time,platform,random,traceback
from pathlib import Path
if not (os.environ.get('ALLOW_LOCAL') or os.environ.get('SLURM_JOB_ID')) and ('head' in socket.gethostname().lower() or socket.gethostname().lower()=='bcmnode1'):
    raise SystemExit('Slurm compute allocation required; head-node execution forbidden')
import numpy as np
import torch
import monai,scipy
from monai.inferers import sliding_window_inference
from reused_components import Section3AblationUNet,compute_metrics_pair
from reused_preprocessing import get_preproc_transforms

HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'config.json').read_text())
IDS=json.loads((HERE/'official_test_ids.json').read_text())
OUT=HERE/'outputs';OUT.mkdir(exist_ok=True)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(name,data):
    p=OUT/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(p)
def environment():
    return dict(host=socket.gethostname(),slurm_job=os.environ['SLURM_JOB_ID'],task=os.environ.get('SLURM_ARRAY_TASK_ID'),
      python=sys.version,platform=platform.platform(),torch=torch.__version__,monai=monai.__version__,numpy=np.__version__,
      scipy=scipy.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
      gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
def frozen():
    lock=json.loads((HERE/'freeze.json').read_text())
    for name,digest in lock['files'].items():assert sha(HERE/name)==digest, 'Frozen source/config changed: '+name
    return lock
def configure():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
def checkpoint(target,lock):
    p=Path(target['path']);assert sha(p)==lock['checkpoints'][target['name']]['sha256'],'Checkpoint changed'
    ck=torch.load(p,map_location='cpu',weights_only=True)
    assert ck['epoch']==500 and ck['seed']==target['seed'] and ck['exp']==target['exp']
    assert ck['in_channels']==5 and ck['backbone']=='resnet50'
    model=Section3AblationUNet(in_channels=5,out_channels=1,backbone='resnet50')
    model.load_state_dict(ck['model_state_dict'],strict=True)
    assert sum(p.numel() for p in model.parameters())==ck['total_params']
    del ck
    return model.eval()
def metrics(pred,gt):
    ps=int(pred.sum());gs=int(gt.sum())
    if ps and gs:
        d,i,p,r,h,a=compute_metrics_pair(pred,gt,spacing=tuple(CFG['metrics']['spacing']))
    elif not ps and not gs:d=i=100.;p=r=100.;h=a=0.
    else:d=i=0.;p=0. if ps else float('nan');r=0. if gs else float('nan');h=a=float('nan')
    return dict(dice=d,iou=i,precision=p,recall=r,hd95=h,assd=a,
      empty_prediction=int(ps==0),empty_reference=int(gs==0),surface_defined=int((ps>0 and gs>0) or (ps==gs==0)))
def self_checks():
    a=np.zeros((7,7,7),dtype=np.uint8);a[2,2,2]=1;b=np.zeros_like(a);b[3,2,2]=1
    m=metrics(a,a);assert m['dice']==100 and m['iou']==100 and m['hd95']==0 and m['assd']==0
    m=metrics(a,b);assert m['dice']==0 and abs(m['hd95']-.8)<1e-5 and abs(m['assd']-.8)<1e-5
    z=torch.tensor([-1.,0.,.49,.5,1.]);assert (torch.sigmoid(z)>=.5).tolist()==[False,True,True,True,True]
    assert (z>=.5).tolist()==[False,False,False,True,True]
    assert np.isnan(metrics(a,np.zeros_like(a))['hd95'])
def preflight():
    lock=frozen();configure();self_checks()
    assert len(IDS)==len(set(IDS))==306
    root=Path(CFG['data_root']);cache=Path(CFG['cache_root'])
    assert sha(root/'train_test_splits.csv')==lock['official_split_sha256']
    with (root/'train_test_splits.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    official=sorted(r['test_split'] for r in rows if r['test_split']);train={r['train_split'] for r in rows if r['train_split']}
    assert IDS==official and not set(IDS)&train
    actual={x.name[:-5] for x in cache.glob('*_test') if x.is_dir()};assert actual==set(IDS),'Test cache missing/extra patients'
    # This phase uses CPU only. The inference entry point separately enforces GPU/bfloat16 availability.
    for target in CFG['targets']:
        m=checkpoint(target,lock);del m
    manifest=[]
    for index,pid in enumerate(IDS):
        files={k:cache/f'{pid}_test'/(k+'.npy') for k in ['image','label']}
        im=np.load(files['image'],mmap_mode='r');gt=np.load(files['label'],mmap_mode='r')
        assert im.dtype==np.float16 and im.ndim==4 and im.shape[0]==5,pid+' image schema'
        assert gt.dtype==np.uint8 and gt.shape==(1,*im.shape[1:]),pid+' label schema'
        assert all(a>=b for a,b in zip(im.shape[1:],CFG['inference']['patch'])),pid+' padding'
        assert np.isfinite(im).all() and np.isin(gt,[0,1]).all(),pid+' finite/binary'
        # Float16 cache was cast after float32 subtraction: tolerate quantisation only.
        for k in [1,2]:
            expected=im[k].astype(np.float32)-im[0].astype(np.float32)
            observed=im[k+2].astype(np.float32)
            bound=.004*(1+np.abs(im[k].astype(np.float32))+np.abs(im[0].astype(np.float32)))
            assert np.all(np.abs(expected-observed)<=bound),pid+' channel ordering/subtraction'
        records={}
        for key,path in files.items():
            st=path.stat();digest=sha(path);st2=path.stat();assert (st.st_size,st.st_mtime_ns)==(st2.st_size,st2.st_mtime_ns)
            records[key]=dict(path=str(path),bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=digest)
        manifest.append(dict(pid=pid,shape=list(im.shape),files=records))
        del im,gt
        if index%30==0:print('PREFLIGHT_CACHE',index+1,len(IDS),flush=True)
    spot=[];tf=get_preproc_transforms(is_train=False)
    for pid in CFG['preprocessing']['raw_reconstruction_ids']:
        paths={f'image_{k}':str(root/'images'/pid/f'{pid}_{k:04d}.nii') for k in [0,1,2]}
        paths['label']=str(root/'segmentations/expert'/f'{pid}.nii')
        raw_hashes={k:sha(p) for k,p in paths.items()};res=tf(paths)
        a,b,c=(res[f'image_{k}'].as_tensor() for k in [0,1,2])
        raw=torch.cat([a,b,c,b-a,c-a],0).numpy().astype(np.float16)
        cached=np.load(cache/f'{pid}_test/image.npy',mmap_mode='r');label=np.load(cache/f'{pid}_test/label.npy',mmap_mode='r')
        expected_label=(res['label'].as_tensor().numpy()>.5).astype(np.uint8)
        assert raw.shape==cached.shape,pid+' raw/cache shape mismatch'
        channel_checks=[]
        for k in range(5):
            difference=np.abs(raw[k].astype(np.float32)-cached[k].astype(np.float32))
            ulp=np.maximum(np.abs(np.spacing(raw[k])).astype(np.float32),np.abs(np.spacing(cached[k])).astype(np.float32))
            bound=CFG['preprocessing']['raw_float16_ulp_allowance']*ulp+CFG['preprocessing']['raw_absolute_roundoff']
            assert np.all(difference<=bound),pid+' raw/cache intensity difference exceeds float16 precision bound'
            channel_checks.append(dict(channel=k,max_absolute_error=float(difference.max()),mean_absolute_error=float(difference.mean()),voxels_exceeding_bound=0))
        assert np.array_equal(expected_label,label),pid+' raw/cache label mismatch'
        affine=res['image_0'].affine.cpu().numpy();spacing=np.linalg.norm(affine[:3,:3],axis=0)
        assert np.allclose(spacing,CFG['metrics']['spacing'],atol=1e-5),pid+' spacing'
        spot.append(dict(pid=pid,raw_file_hashes=raw_hashes,cache_shape=list(raw.shape),spacing=spacing.tolist(),raw_cache_match_within_float16_precision=True,label_exact_match=True,channel_checks=channel_checks))
        del raw,cached,label,expected_label,res,a,b,c
        print('PREFLIGHT_RAW_MATCH',pid,flush=True)
    write('cache_manifest.json',manifest)
    write('preflight_passed.json',dict(status='PASS',freeze_sha256=sha(HERE/'freeze.json'),
      cache_manifest_sha256=sha(OUT/'cache_manifest.json'),environment=environment(),raw_spot_checks=spot,
      historical_training_membership='UNKNOWN',no_test_scores_computed=True,completed_unix=time.time()))
    print('PREFLIGHT PASS',flush=True)
def evaluate(index):
    lock=frozen();configure()
    gate=json.loads((OUT/'preflight_passed.json').read_text());assert gate['status']=='PASS' and gate['freeze_sha256']==sha(HERE/'freeze.json')
    assert gate['cache_manifest_sha256']==sha(OUT/'cache_manifest.json')
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    manifest=json.loads((OUT/'cache_manifest.json').read_text());target=CFG['targets'][index]
    model=checkpoint(target,lock).cuda();output=OUT/(target['name']+'_per_case.csv');assert not output.exists(),'Refuse silent overwrite'
    start=time.time();records=[]
    for i,entry in enumerate(manifest):
        for rec in entry['files'].values():
            st=Path(rec['path']).stat();assert st.st_size==rec['bytes'] and st.st_mtime_ns==rec['mtime_ns'],'Cache changed since preflight'
        image=np.load(entry['files']['image']['path'],mmap_mode='r');label=np.load(entry['files']['label']['path'],mmap_mode='r')[0]>0
        tensor=torch.from_numpy(np.asarray(image,dtype=np.float32)).unsqueeze(0).cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            logits=sliding_window_inference(tensor,tuple(CFG['inference']['patch']),
              sw_batch_size=CFG['inference']['window_batch'],predictor=model,
              overlap=CFG['inference']['overlap'],mode=CFG['inference']['blend'])
        logits=logits[0,0].float();assert torch.isfinite(logits).all(),'Nonfinite logits'
        probability=torch.sigmoid(logits);pred=(probability>=CFG['inference']['threshold']).cpu().numpy()
        row=dict(pid=entry['pid'],centre=entry['pid'].split('_')[0],**metrics(pred,label))
        # Diagnostic only: no parameter changes or selection based on this value.
        row['float32_sigmoid_vs_logit_zero_ties']=int(((probability>=.5)!=(logits>=0)).sum().item())
        records.append(row)
        with output.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(records)
        del tensor,logits,probability,pred,image,label
        if i%30==0:print(target['name'],'CASE',i+1,len(IDS),'SECONDS',round(time.time()-start,1),flush=True)
    assert [r['pid'] for r in records]==IDS
    write(target['name']+'_done.json',dict(status='DONE',cases=len(records),target=target,
      checkpoint_sha256=lock['checkpoints'][target['name']]['sha256'],freeze_sha256=sha(HERE/'freeze.json'),
      per_case_csv_sha256=sha(output),environment=environment(),seconds=time.time()-start,
      historical_training_membership='UNKNOWN',threshold_protocol=CFG['inference'],training_executed=False))
    print('DONE',target['name'],flush=True)
if __name__=='__main__':
    try:
        if sys.argv[1]=='preflight':preflight()
        elif sys.argv[1]=='evaluate':evaluate(int(os.environ['SLURM_ARRAY_TASK_ID']))
        else:raise ValueError('Unknown mode')
    except Exception:
        write('failure_'+sys.argv[1]+'_'+os.environ.get('SLURM_ARRAY_TASK_ID','preflight')+'.json',
          dict(status='FAILED',traceback=traceback.format_exc(),environment=environment(),time=time.time()))
        raise
