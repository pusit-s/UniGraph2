"""Focused correctness checks for graph attention, masking, SPD and multimodal fusion."""
import itertools
import unittest
import numpy as np
import torch
from scipy import sparse
from graph_data import fuse_features, spd_targets, Graph, ppr_nodes
from model import GATLayer, UniGraph2

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
                torch.testing.assert_close(model.encode(x,edges),model.encode(x,edges))
                before=model.encode(x,edges)
                model.mask_token.add_(100)
                torch.testing.assert_close(before,model.encode(x,edges))
                perm=torch.tensor([2,0,1],device=device)
                inverse=torch.argsort(perm)
                torch.testing.assert_close(before[perm],model.encode(x[perm],inverse[edges]))

if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
