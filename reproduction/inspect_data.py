import sys, importlib, torch
from pathlib import Path
print('python',sys.version,'torch',torch.__version__,'cuda',torch.cuda.is_available(), flush=True)
for name in ['scipy','torch_geometric','dgl']:
    try:
        m=importlib.import_module(name); print(name,m.__version__)
    except Exception as e: print(name,repr(e))
def summary(x, depth=0):
    if isinstance(x,torch.Tensor): return {'shape':list(x.shape),'dtype':str(x.dtype),'min':x.min().item() if x.numel() else None,'max':x.max().item() if x.numel() else None}
    if isinstance(x,dict): return {'length':len(x), 'sample':{str(k):summary(v,depth+1) for k,v in list(x.items())[:8]}}
    if isinstance(x,(tuple,list)): return [summary(v,depth+1) for v in x[:3]]
    return str(x)[:80]
root=Path('/home/pusit.see/dataset/raw/paper_2602_00067/mm_graph')
for name in ['ele-fashion','cloth-copurchase','sports-copurchase']:
    for fname in ['clip_feat.pt','labels-w-missing.pt','nc_edges-nodeid.pt','split.pt','lp-edge-split.pt','node_mapping.pt']:
        path=root/name/fname
        if path.exists():
            x=torch.load(path,map_location='cpu',weights_only=False)
            print(name,fname,summary(x),flush=True)
