"""Pretrain UniGraph2 on local MM-Graph domains and evaluate a frozen NC/LP encoder."""
import argparse
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from graph_data import read_graph, ppr_nodes, spd_targets
from model import UniGraph2
from roma import posttrain
from link_probe import prepare_link_splits, lp_probe


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def probe(embeddings, y, splits, seed, epochs=5000, patience=100):
    seed_all(seed)
    x = F.normalize(embeddings.detach(), dim=-1)
    indices = {}
    for name, aliases in {'train':['train','train_idx'], 'valid':['valid','val','val_idx'], 'test':['test','test_idx']}.items():
        key = next(k for k in aliases if k in splits)
        idx = torch.as_tensor(splits[key], dtype=torch.long, device=x.device)
        indices[name] = idx[y[idx] >= 0]
        if not indices[name].numel():
            raise ValueError(f'empty labeled {name} split')
    for a,b in [('train','valid'),('train','test'),('valid','test')]:
        if set(indices[a].tolist()) & set(indices[b].tolist()):
            raise ValueError('overlapping evaluation splits')
    head = nn.Linear(x.shape[1], int(y[y>=0].max())+1).to(x.device)
    opt = torch.optim.Adam(head.parameters(), lr=.01)
    best, state, selected, stale = -1., None, 0, 0
    for epoch in range(1, epochs+1):
        head.train()
        logits = head(x[indices['train']])
        loss = F.cross_entropy(logits, y[indices['train']])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            validation = (head(x[indices['valid']]).argmax(-1) == y[indices['valid']]).float().mean().item()
        if validation > best:
            best, state, selected, stale = validation, copy.deepcopy(head.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    head.load_state_dict(state)
    with torch.no_grad():
        test = (head(x[indices['test']]).argmax(-1) == y[indices['test']]).float().mean().item()
    return {'valid_accuracy':best, 'test_accuracy':test, 'selected_epoch':selected,
            'epochs_run':epoch, 'split_sizes':{k:len(v) for k,v in indices.items()}}, head.state_dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/home/pusit.see/dataset/raw/paper_2602_00067/mm_graph'))
    parser.add_argument('--pretrain', nargs='+', default=['cloth-copurchase'])
    parser.add_argument('--evaluate', default='ele-fashion')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--experts', type=int, default=8)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=0, help='0 uses dense routing from paper equations; 2 matches upstream sparse routing')
    parser.add_argument('--expert', choices=['mlp','attention'], default='mlp', help='MoE expert type: 2-layer MLP (paper) or single transformer attention head over the nodes of each graph')
    parser.add_argument('--router-posttrain', choices=['none', 'task', 'roma'], default='none',
                        help='optional router-only post-training; task is the no-alignment control')
    parser.add_argument('--resume-pretrain', type=Path, help='completed baseline run to branch from without repeating pre-training')
    parser.add_argument('--router-epochs', type=int, default=3)
    parser.add_argument('--router-lr', type=float, default=7e-5)
    parser.add_argument('--roma-lambda', type=float, default=.8)
    parser.add_argument('--roma-k', type=int, default=5)
    parser.add_argument('--roma-sigma', type=float, default=.4)
    parser.add_argument('--roma-success-fraction', type=float, default=.5)
    parser.add_argument('--dropout', type=float, default=.4)
    parser.add_argument('--mask-rate', type=float, default=.8)
    parser.add_argument('--gamma', type=float, default=2.)
    parser.add_argument('--lambda-spd', type=float, default=.1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--steps', type=int, default=32, help='sampled subgraphs per domain per epoch, NOT full dataset epochs')
    parser.add_argument('--subgraph-nodes', type=int, default=256)
    parser.add_argument('--spd-pairs', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-4, help='official config value; paper Table 9 lists 1e-3, which collapses here')
    parser.add_argument('--warmup', type=float, default=.1, help='fraction of steps with linear warmup, then cosine decay; 0 is constant lr as in the paper')
    parser.add_argument('--standardize', action=argparse.BooleanOptionalAction, default=True, help='standardize encoder input with pre-training-domain statistics (not in the paper)')
    parser.add_argument('--weight-decay', type=float, default=.01)
    parser.add_argument('--eval-limit', type=int, default=4096, help='first N nodes induced graph; 0 for full graph')
    parser.add_argument('--probe-epochs', type=int, default=5000)
    parser.add_argument('--probe-patience', type=int, default=100)
    parser.add_argument('--lp-negatives', type=int, default=100, help='fixed non-neighbors/query in bounded link evaluation; not official full-graph negatives')
    parser.add_argument('--lp-train-edges', type=int, default=4096)
    parser.add_argument('--lp-eval-edges', type=int, default=1024, help='maximum positive queries in each validation/test split')
    parser.add_argument('--lp-split-seed', type=int, default=1729, help='fixed cohort/candidates, shared across encoder and probe seeds')
    parser.add_argument('--lp-validation-every', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--probe-seeds', type=int, nargs='+', default=[0,1,2])
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    args = parser.parse_args()
    if min(args.epochs,args.steps,args.subgraph_nodes,args.probe_epochs,args.probe_patience)<1 or args.eval_limit<0 or args.spd_pairs<0 or not 0<=args.warmup<1:
        parser.error('invalid run limits')
    if min(args.lp_negatives,args.lp_train_edges,args.lp_eval_edges,args.lp_validation_every) < 1:
        parser.error('invalid link-probe limits')
    if args.evaluate in args.pretrain:
        parser.error('evaluation domain must be excluded from pre-training')
    if args.router_epochs < 1 or args.router_lr <= 0 or args.roma_lambda < 0 or args.roma_k < 1 or args.roma_sigma <= 0 or not 0 < args.roma_success_fraction <= 1:
        parser.error('invalid router post-training configuration')
    if args.router_posttrain != 'none' and (args.top_k == 1 or args.experts < 2):
        parser.error('router tuning requires at least two active experts for differentiable routing')
    if args.resume_pretrain and (args.router_posttrain == 'none' or args.resume_pretrain.resolve() == args.output.resolve()):
        parser.error('resume into a separate output directory with task or roma post-training')
    if (args.output/'results.json').exists():
        parser.error('output already contains results; choose a new directory')
    args.output.mkdir(parents=True, exist_ok=True)
    source_hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    torch.set_num_threads(4)
    seed_all(args.seed)
    started = time.time()
    graphs = [read_graph(args.data_root, name) for name in args.pretrain]
    target = read_graph(args.data_root, args.evaluate, limit=args.eval_limit, load_edge_splits=True)
    task = 'node_classification' if target.y is not None else 'link_prediction'
    primary_metric = 'accuracy' if target.y is not None else 'mrr'
    evaluation_protocol = {'task': task, 'primary_metric': primary_metric, 'nodes': len(target.x)}
    link_splits = None
    if target.edge_splits is not None:
        link_splits, evaluation_protocol = prepare_link_splits(
            target.edge_splits, len(target.x), args.lp_split_seed, args.lp_negatives,
            args.lp_train_edges, args.lp_eval_edges)
        torch.save(link_splits, args.output/'evaluation_splits.pt')
    print('EVALUATION', evaluation_protocol, flush=True)
    if len({g.x.shape[1] for g in graphs+[target]}) != 1:
        raise ValueError('all domains must use the same CLIP feature space')
    model_kwargs = dict(input_dim=graphs[0].x.shape[1], domains=args.pretrain, hidden=args.hidden,
                        layers=args.layers, experts=args.experts, heads=args.heads, top_k=args.top_k, expert=args.expert,
                        dropout=args.dropout, mask_rate=args.mask_rate, gamma=args.gamma, lambda_spd=args.lambda_spd)
    model = UniGraph2(**model_kwargs).to(args.device)
    if args.standardize:
        features = torch.cat([g.x for g in graphs])
        model.feature_mean.copy_(features.mean(0))
        model.feature_std.copy_(features.std(0).clamp_min(1e-6))
    tx, te, _ = target.subgraph(np.arange(len(target.x)))
    tx, te = tx.to(args.device), te.to(args.device)
    y = target.y.to(args.device) if target.y is not None else None
    print('DATA',[(g.name,len(g.x),g.x.shape[1],g.adjacency.nnz) for g in graphs+[target]], flush=True)
    print('MODEL',model_kwargs, 'parameters',sum(p.numel() for p in model.parameters()),flush=True)
    model.eval()
    with torch.no_grad():
        initial = model.encode(tx, te).detach().cpu()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total = args.epochs * args.steps * len(graphs)
    warm = max(1, int(args.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1., (t+1)/warm) * .5*(1+np.cos(np.pi*t/total)) if args.warmup else 1.)
    history = []
    rng = np.random.default_rng(args.seed)
    candidates = [np.flatnonzero(np.diff(g.adjacency.indptr)>0) for g in graphs]
    # Cache a fixed pool of PPR subgraphs; reshuffle and remask each epoch.
    pools = []
    for graph, eligible in zip(graphs,candidates):
        if len(eligible)==0:
            raise ValueError(f'{graph.name} has no non-isolated nodes')
        pool=[]
        for seed in rng.choice(eligible, args.steps, replace=len(eligible)<args.steps):
            ids=ppr_nodes(graph,int(seed),args.subgraph_nodes)
            if len(ids)<2:
                raise ValueError('PPR returned fewer than two nodes')
            x, edges, adj=graph.subgraph(ids)
            pairs, distances=spd_targets(adj,args.spd_pairs,rng)
            pool.append((x,edges,pairs,distances,ids))
        pools.append(pool)
        print('PPR',graph.name,'pool',len(pool),'nodes',[len(item[0]) for item in pool],flush=True)
    torch.save({g.name:[item[4] for item in pool] for g,pool in zip(graphs,pools)},args.output/'sampled_nodes.pt')
    lineage = None
    if args.resume_pretrain:
        baseline = json.loads((args.resume_pretrain/'results.json').read_text())
        if baseline.get('status') != 'completed' or baseline['args'].get('router_posttrain', 'none') != 'none':
            raise ValueError('resume requires a completed baseline without router post-training')
        base_keys = ['pretrain', 'evaluate', 'hidden', 'layers', 'experts', 'heads', 'top_k', 'expert',
                     'dropout', 'mask_rate', 'gamma', 'lambda_spd', 'epochs', 'steps', 'subgraph_nodes',
                     'spd_pairs', 'lr', 'warmup', 'standardize', 'weight_decay', 'eval_limit', 'seed',
                     'probe_epochs', 'probe_patience', 'probe_seeds', 'lp_negatives', 'lp_train_edges',
                     'lp_eval_edges', 'lp_split_seed', 'lp_validation_every']
        for key in base_keys:
            if baseline['args'].get(key) != getattr(args, key):
                raise ValueError(f'baseline configuration mismatch: {key}')
        if Path(baseline['args']['data_root']).resolve() != args.data_root.resolve():
            raise ValueError('baseline data root mismatch')
        for path, metadata in baseline['data_files'].items():
            if Path(path).stat().st_size != metadata['bytes'] or Path(path).stat().st_mtime_ns != metadata['mtime_ns']:
                raise ValueError(f'baseline data changed: {path}')
        # Model and pool code must match. Analysis/plotting files may be added later.
        for name in ['model.py', 'graph_data.py']:
            if baseline['source_sha256'][name] != source_hashes[name]:
                raise ValueError(f'baseline source mismatch: {name}')
        saved_nodes = torch.load(args.resume_pretrain/'sampled_nodes.pt', weights_only=False)
        for graph, pool in zip(graphs, pools):
            if len(saved_nodes[graph.name]) != len(pool) or any(not np.array_equal(ids, item[4]) for ids, item in zip(saved_nodes[graph.name], pool)):
                raise ValueError('baseline sampled node pool mismatch')
        checkpoint_path = args.resume_pretrain/'checkpoint.pt'
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        if checkpoint['epoch'] != args.epochs or checkpoint['model_kwargs'] != model_kwargs:
            raise ValueError('baseline checkpoint configuration mismatch')
        model.load_state_dict(checkpoint['model'])
        history = baseline['history']
        baseline_initial = torch.load(args.resume_pretrain/'embeddings.pt', weights_only=False)['untrained']
        # CUDA scatter reductions can differ by a few ulps across processes.
        # Reuse the saved baseline representation for exactly paired probes.
        torch.testing.assert_close(initial, baseline_initial, rtol=1e-4, atol=5e-6)
        initial = baseline_initial
        lineage = {'run': str(args.resume_pretrain.resolve()),
                   'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                   'source_sha256': baseline['source_sha256'],
                   'slurm_job_id': baseline['environment']['slurm_job_id']}
        (args.output/'history.json').write_text(json.dumps(history, indent=2))
    for epoch in ([] if args.resume_pretrain else range(1,args.epochs+1)):
        model.train()
        metrics=[]
        schedule=[(i,j) for i,pool in enumerate(pools) for j in range(len(pool))]
        rng.shuffle(schedule)
        for i,j in schedule:
            x,edges,pairs,distances,_=pools[i][j]
            opt.zero_grad(set_to_none=True)
            loss, detail=model.objective(x.to(args.device),edges.to(args.device),graphs[i].name,
                                        pairs.to(args.device),distances.to(args.device))
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite loss: {detail}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            sched.step()
            metrics.append(detail)
        row={'epoch':epoch,**{k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]}}
        history.append(row)
        print('TRAIN',row,flush=True)
        (args.output/'history.json').write_text(json.dumps(history,indent=2))
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'model_kwargs':model_kwargs,'epoch':epoch},args.output/'checkpoint.pt')
    router_history = []
    if args.router_posttrain != 'none':
        before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        router_history = posttrain(model, graphs, pools, args, args.output)
        changed = [key for key, value in model.state_dict().items() if not torch.equal(value.cpu(), before[key])]
        if not changed or any(not key.startswith('moe.gate.') for key in changed):
            raise RuntimeError(f'router-only update invariant failed: {changed}')
        (args.output/'router_history.json').write_text(json.dumps(router_history, indent=2))
        torch.save({'model': model.state_dict(), 'model_kwargs': model_kwargs,
                    'router_history': router_history, 'changed_parameters': changed}, args.output/'router_checkpoint.pt')
    model.eval()
    with torch.no_grad():
        learned=model.encode(tx,te).detach().cpu()
        routing=model.routing_weights(tx).detach().cpu()
    torch.save({'pretrained':learned,'untrained':initial,'routing':routing,
                'node_ids':torch.arange(len(tx))},args.output/'embeddings.pt')
    results={}
    for name, emb in [('CLIP',tx.cpu()),('NoPretrain',initial),('UniGraph2',learned)]:
        runs=[]
        for probe_seed in args.probe_seeds:
            if task == 'node_classification':
                metrics,head=probe(emb.to(args.device),y,target.splits,probe_seed,args.probe_epochs,args.probe_patience)
            else:
                metrics,head=lp_probe(emb.to(args.device),link_splits,probe_seed,args.probe_epochs,args.probe_patience,args.lp_validation_every)
            metrics['seed']=probe_seed
            runs.append(metrics)
            torch.save(head,args.output/f'probe_{name}_{probe_seed}.pt')
            print('PROBE',name,metrics,flush=True)
        scores=[r[f'test_{primary_metric}']*100 for r in runs]
        results[name]={f'mean_test_{primary_metric}_percent':float(np.mean(scores)),
                       f'std_test_{primary_metric}_percent':float(np.std(scores)), 'runs':runs}
    report={'status':'completed','scope':'bounded real-data method reproduction; not full paper benchmark',
            'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'model_kwargs':model_kwargs,'history':history,'router_history':router_history,
            'pretrain_lineage':lineage,'results':results,'seconds':time.time()-started,
            'task':task,'primary_metric':primary_metric,'evaluation_protocol':evaluation_protocol,
            'parameter_counts':{'total':sum(p.numel() for p in model.parameters()),
                                'router':sum(p.numel() for p in model.moe.gate.parameters())},
            'environment':{'python':os.sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
                           'gpu':torch.cuda.get_device_name() if args.device=='cuda' else None,
                           'slurm_job_id':os.environ.get('SLURM_JOB_ID')},
            'graphs':{g.name:{'nodes':len(g.x),'directed_edges':g.adjacency.nnz,'feature_dim':g.x.shape[1]} for g in graphs+[target]}}
    report['source_sha256'] = source_hashes
    report['data_files'] = {str(p):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
                            for name in set(args.pretrain+[args.evaluate])
                            for p in (args.data_root/name).glob('*.pt')}
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    print('COMPLETE',results,flush=True)

if __name__=='__main__':
    main()
