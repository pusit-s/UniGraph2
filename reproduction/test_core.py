"""Focused correctness checks for graph attention, masking, SPD and multimodal fusion."""
import itertools
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from scipy import sparse
from graph_data import fuse_features, spd_targets, Graph, ppr_nodes
from model import GATLayer, UniGraph2
from roma import build_bank, freeze_for_router, neighbor_targets, manifold_loss
from link_probe import prepare_link_splits, ranking_metrics, lp_probe

class CoreTests(unittest.TestCase):
    def test_available_modalities(self):
        text=torch.tensor([[2.,4.],[3.,5.]])
        image=torch.tensor([[4.,6.],[0.,0.]])
        torch.testing.assert_close(fuse_features([text,image]),torch.tensor([[3.,5.],[3.,5.]]))
        with self.assertRaises(ValueError):
            fuse_features([torch.zeros(2,2)])

    def test_spd_disconnected(self):
        adj=sparse.csr_matrix(np.array([[0,1,0],[1,0,0],[0,0,0]]))
        pairs,dist=spd_targets(adj,0,np.random.default_rng(0))
        self.assertTrue(torch.isfinite(dist).all())
        self.assertEqual(len(dist),5)
        self.assertEqual(dist.max().item(),1)
        graph=Graph('tiny',torch.ones(3,2),adj)
        self.assertEqual(set(ppr_nodes(graph,0,3).tolist()),{0,1})

    def test_attention_matches_dense(self):
        torch.manual_seed(0)
        layer=GATLayer(8,2,0).eval()
        x=torch.randn(3,8)
        edges=torch.tensor([[0,1,2,0,1],[0,1,2,1,0]])
        h=layer.linear(x).view(3,2,4)
        src,dst=edges
        scores=torch.nn.functional.leaky_relu((h*layer.a_src).sum(-1)[src]+(h*layer.a_dst).sum(-1)[dst],.2)
        ref=torch.zeros_like(h)
        for i in range(3):
            keep=dst==i
            ref[i]=(scores[keep].softmax(0)[...,None]*h[src[keep]]).sum(0)
        ref=torch.nn.functional.elu(layer.norm(ref.reshape(3,8)+layer.bias+x))
        torch.testing.assert_close(layer(x,edges),ref)

    def test_input_standardization(self):
        torch.manual_seed(0)
        model=UniGraph2(6,['a'],hidden=8,experts=2,layers=1,heads=2,dropout=0).eval()
        x=torch.randn(3,6)*.05+.3
        edges=torch.tensor([[0,1,2],[0,1,2]])
        expected=model.encode((x-x.mean(0))/x.std(0),edges)
        model.feature_mean.copy_(x.mean(0))
        model.feature_std.copy_(x.std(0))
        torch.testing.assert_close(model.encode(x,edges),expected)
        before=x.clone()
        model.encode(x,edges,torch.tensor([0]))
        torch.testing.assert_close(x,before)

    def test_inference_and_gradients(self):
        devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
        for device,expert,top_k in itertools.product(devices,['mlp','attention'],[0,1]):
            torch.manual_seed(0)
            model=UniGraph2(6,['a','b'],hidden=8,experts=2,layers=2,heads=2,dropout=0,top_k=top_k,expert=expert).to(device)
            x=torch.randn(3,6,device=device)
            edges=torch.tensor([[0,1,2,0,1],[0,1,2,1,0]],device=device)
            pairs=torch.tensor([[0,1],[1,0]],device=device)
            distances=torch.ones(2,device=device)
            loss,_=model.objective(x,edges,'a',pairs,distances)
            loss.backward()
            for module in [model.moe,model.gnn,model.decoders['a'],model.spd]:
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()))
            self.assertTrue(model.mask_token.grad.abs().sum()>0)
            self.assertTrue(all(p.grad is None for p in model.decoders['b'].parameters()))
            model.eval()
            with torch.no_grad():
                routes=model.routing_weights(x)
                torch.testing.assert_close(routes.sum(-1),torch.ones(len(x),device=device))
                self.assertTrue((routes>=0).all())
                self.assertTrue(((routes>0).sum(-1)==(top_k or 2)).all())
                torch.testing.assert_close(model.encode(x,edges),model.encode(x,edges))
                before=model.encode(x,edges)
                model.mask_token.add_(100)
                torch.testing.assert_close(before,model.encode(x,edges))
                perm=torch.tensor([2,0,1],device=device)
                inverse=torch.argsort(perm)
                torch.testing.assert_close(before[perm],model.encode(x[perm],inverse[edges]))
                torch.testing.assert_close(routes[perm],model.routing_weights(x[perm]))

    def test_manifold_formula_and_self_exclusion(self):
        bank={'features':torch.tensor([[1.,0.],[0.,1.],[-1.,0.]]),
              'routes':torch.tensor([[.9,.1],[.2,.8],[.6,.4]],requires_grad=True),
              'identities':torch.tensor([[0,1],[0,2],[1,1]])}
        x=torch.tensor([[1.,0.]])
        neighbors,weights=neighbor_targets(x,torch.tensor([[0,1]]),bank,k=3,sigma=1.)
        # Squared distances are 2 and 4, hence Gaussian logits -1 and -2.
        torch.testing.assert_close(weights[0,:2],torch.tensor([-1.,-2.]).softmax(-1))
        self.assertEqual(weights[0,2].item(),0.)
        routes=torch.tensor([[.4,.6]],requires_grad=True)
        loss=manifold_loss(routes,neighbors,weights)
        expected=weights[0,0]*.08+weights[0,1]*.08
        torch.testing.assert_close(loss,expected)
        loss.backward()
        self.assertGreater(routes.grad.abs().sum().item(),0)
        self.assertIsNone(bank['routes'].grad)
        # Domain ID participates in identity: domain 1/node 1 is a valid neighbor.
        self.assertTrue((neighbors[0,1]==bank['routes'][2]).all())

    def test_router_only_alignment_gradients(self):
        devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
        for device,expert,top_k in itertools.product(devices,['mlp','attention'],[0,2]):
            torch.manual_seed(5)
            model=UniGraph2(6,['a','b'],hidden=8,experts=3,layers=1,heads=2,dropout=.4,expert=expert,top_k=top_k).to(device)
            model.feature_mean.fill_(.2)
            model.feature_std.fill_(.5)
            x=torch.randn(6,6,device=device)
            edges=torch.arange(6,device=device).repeat(2,1)
            pairs=torch.tensor([[0,1],[1,0]],device=device)
            before={key:value.clone() for key,value in model.state_dict().items()}
            parameters=freeze_for_router(model)
            self.assertFalse(model.training)
            # A nonconstant, frozen teacher makes a detached-online-router bug fail.
            teacher=torch.softmax(torch.randn(6,2,3,device=device),-1).requires_grad_()
            weights=torch.full((6,2),.5,device=device)
            loss=manifold_loss(model.routing_weights(x),teacher,weights)
            loss.backward()
            self.assertGreater(sum(p.grad.abs().sum().item() for p in parameters),0)
            self.assertIsNone(teacher.grad)
            self.assertTrue(all(p.grad is None for name,p in model.named_parameters() if not name.startswith('moe.gate.')))
            optimizer=torch.optim.SGD(parameters,lr=.1)
            optimizer.step()
            changed=[key for key,value in model.state_dict().items() if not torch.equal(value,before[key])]
            self.assertTrue(changed)
            self.assertTrue(all(key.startswith('moe.gate.') for key in changed))
            optimizer.zero_grad(set_to_none=True)
            task,_=model.objective(x,edges,'a',pairs,torch.ones(2,device=device))
            task.backward()
            self.assertGreater(sum(p.grad.abs().sum().item() for p in parameters),0)

    def test_bank_is_label_free_deduplicated_and_rng_isolated(self):
        torch.manual_seed(0)
        graph=Graph('a',torch.randn(6,6),sparse.eye(6,format='csr'))
        x,edges,adj=graph.subgraph(np.arange(6))
        pairs,distances=spd_targets(adj,0,np.random.default_rng(0))
        item=(x,edges,pairs,distances,np.arange(6))
        model=UniGraph2(6,['a'],hidden=8,experts=2,layers=1,heads=2,dropout=0).eval()
        state=torch.random.get_rng_state().clone()
        bank=build_bank(model,[graph],[[item,item]],fraction=.5,seed=7)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        self.assertEqual(len(bank['identities'].unique(dim=0)),len(bank['identities']))
        self.assertFalse(bank['routes'].requires_grad)
        # No labels/splits exist on this graph, and the bank still builds.
        self.assertIsNone(graph.y)
        torch.testing.assert_close(bank['routes'],model.routing_weights(graph.x[bank['identities'][:,1]]))

    def test_comparison_requires_complete_paired_runs(self):
        from plot_roma import CONDITIONS, review_runs, summarize
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for seed in [0,1,2]:
                for expert,method,_ in CONDITIONS:
                    path=root/f'seed-{seed}'/f'{expert}-{method}'
                    path.mkdir(parents=True)
                    (path/'checkpoint.pt').write_bytes(b'baseline checkpoint')
                    result={'mean_test_accuracy_percent':float(10+seed*10),
                            'runs':[{'seed':0,'test_accuracy':(10+seed*10)/100}]}
                    report={'status':'completed',
                            'args':{'seed':seed,'expert':expert,'router_posttrain':method,'output':str(path),
                                    'evaluate':'target','pretrain':['source'],'probe_seeds':[0],'lr':1e-4},
                            'results':{name:copy.deepcopy(result) for name in ['CLIP','NoPretrain','UniGraph2']},
                            'graphs':{},'data_files':{},'source_sha256':{name:'fixed' for name in ['model.py','graph_data.py','run.py','roma.py']},
                            'environment':{'slurm_job_id':f'100_{seed}'},
                            'pretrain_lineage':{'run':str(root/f'seed-{seed}'/f'{expert}-none'),
                                                'checkpoint_sha256':hashlib.sha256(b'baseline checkpoint').hexdigest()}}
                    (path/'results.json').write_text(json.dumps(report))
            records=review_runs(root,[0,1,2])
            row=summarize(records,[0,1,2])[0]
            self.assertEqual(row['mean_test_accuracy_percent'],20.)
            self.assertAlmostEqual(row['std_across_encoder_seeds'],np.std([10.,20.,30.]))
            path=root/'seed-1'/'mlp-roma'/'results.json'
            valid=json.loads(path.read_text())
            for mutate in [lambda r:r.update(status='running'),
                           lambda r:r['args'].update(lr=1e-3),
                           lambda r:r['pretrain_lineage'].update(checkpoint_sha256='wrong')]:
                broken=copy.deepcopy(valid)
                mutate(broken)
                path.write_text(json.dumps(broken))
                with self.assertRaises(ValueError):
                    review_runs(root,[0,1,2])
            path.write_text(json.dumps(valid))
            # The same validation/aggregation contract must work for MRR, and
            # must reject a changed negative-candidate population.
            for result_path in root.glob('seed-*/*/results.json'):
                report=json.loads(result_path.read_text())
                report['primary_metric']='mrr'
                report['evaluation_protocol']={'split_sha256':'fixed-candidates'}
                for result in report['results'].values():
                    result['mean_test_mrr_percent']=result.pop('mean_test_accuracy_percent')
                    for probe in result['runs']:
                        probe['test_mrr']=probe.pop('test_accuracy')
                result_path.write_text(json.dumps(report))
            rows=summarize(review_runs(root,[0,1,2]),[0,1,2])
            self.assertEqual(rows[0]['mean_test_mrr_percent'],20.)
            mixed=json.loads(path.read_text())
            mixed['evaluation_protocol']['split_sha256']='different-candidates'
            path.write_text(json.dumps(mixed))
            with self.assertRaises(ValueError):
                review_runs(root,[0,1,2])

    @staticmethod
    def link_fixture():
        return {name:torch.tensor(pairs).T for name,pairs in {
            'train':[(0,1),(1,0),(2,3),(4,5)],
            'valid':[(1,0),(0,2),(6,7),(0,2)],
            'test':[(2,0),(8,9),(10,11)]}.items()}

    def test_link_split_isolation_and_negatives(self):
        raw=self.link_fixture()
        state=torch.random.get_rng_state().clone()
        splits,protocol=prepare_link_splits(raw,20,negatives=3)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        self.assertEqual([len(split['positive'][0]) for split in splits.values()],[3,2,2])
        self.assertEqual(protocol['cohort_counts']['valid']['overlap_with_earlier_splits_removed'],1)
        self.assertEqual(protocol['cohort_counts']['test']['overlap_with_earlier_splits_removed'],1)
        known={tuple(sorted(pair)) for edges in raw.values() for pair in edges.T.tolist()}
        observed=set()
        for name,split in splits.items():
            positives={tuple(sorted(pair)) for pair in split['positive'].T.tolist()}
            self.assertFalse(observed&positives)
            observed|=positives
            for (source,target),negative in zip(split['positive'].T.tolist(),split['negative_targets'].tolist()):
                self.assertNotIn(source,negative)
                self.assertNotIn(target,negative)
                self.assertEqual(len(set(negative)),len(negative))
                self.assertTrue(all(tuple(sorted((source,node))) not in known for node in negative))
        repeated,again=prepare_link_splits(raw,20,negatives=3)
        self.assertEqual(protocol,again)
        _,changed=prepare_link_splits(raw,20,seed=2,negatives=3)
        self.assertNotEqual(protocol['split_sha256'],changed['split_sha256'])

    def test_link_ranking_ties(self):
        scores=torch.tensor([[.8,.9,.7,.8],[.5,.5,.1,.0]])
        result=ranking_metrics(scores)
        self.assertAlmostEqual(result['mrr'],(.4+2/3)/2,places=6)
        self.assertEqual(result['hits@1'],0.)
        self.assertEqual(result['hits@3'],1.)
        with self.assertRaises(ValueError):
            ranking_metrics(torch.tensor([[float('nan'),0.]]))

    def test_link_probe_freezes_embeddings(self):
        splits,_=prepare_link_splits(self.link_fixture(),20,negatives=3)
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            embeddings=torch.randn(20,8,device=device,requires_grad=True)
            first,state=lp_probe(embeddings,splits,0,epochs=10,patience=5,validation_every=2)
            again,_=lp_probe(embeddings,splits,0,epochs=10,patience=5,validation_every=2)
            self.assertEqual(first,again)
            self.assertIsNone(embeddings.grad)
            self.assertTrue(1<=first['selected_epoch']<=first['epochs_run']<=10)
            self.assertTrue(0<first['test_mrr']<=1)

    def test_memory_mapped_bounded_graph_adapter(self):
        from graph_data import read_graph
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)/'tiny'
            folder.mkdir()
            torch.save(torch.ones(20,1024),folder/'clip_feat.pt')
            raw={name:{'source_node':edges[0],'target_node':edges[1]} for name,edges in self.link_fixture().items()}
            torch.save(raw,folder/'lp-edge-split.pt')
            graph=read_graph(temporary,'tiny',limit=12,load_edge_splits=True)
            self.assertEqual(graph.x.shape,(12,512))
            self.assertTrue(set(graph.edge_splits)=={'train','valid','test'})
            # Held-out edge 6--7 must not appear in encoder adjacency.
            self.assertEqual(graph.adjacency[6,7],0)
            self.assertEqual(graph.adjacency[0,1],1)

if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
