"""Inspect saved embedding variation without retraining or changing test selection."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('run',type=Path)
args=parser.parse_args()
torch.set_num_threads(4)
embeddings=torch.load(args.run/'embeddings.pt',map_location='cpu',weights_only=False)
stats={}
for name in ['untrained','pretrained']:
    x=embeddings[name].float()
    normalized=F.normalize(x,dim=-1)
    centered=normalized-normalized.mean(0)
    singular=torch.linalg.svdvals(centered)
    power=singular.square()
    probabilities=power/power.sum().clamp_min(1e-20)
    stats[name]={'finite':bool(torch.isfinite(x).all()),
                 'mean_norm':x.norm(dim=1).mean().item(),
                 'mean_dimension_std':x.std(dim=0).mean().item(),
                 'mean_normalized_dimension_std':normalized.std(dim=0).mean().item(),
                 'mean_cosine_to_centroid':F.cosine_similarity(normalized,normalized.mean(0)[None]).mean().item(),
                 'effective_rank':(-(probabilities*probabilities.clamp_min(1e-20).log()).sum()).exp().item(),
                 'top_component_variance_fraction':probabilities[0].item()}
report={'embedding_diagnostics':stats,
        'interpretation':'High cosine-to-centroid and low normalized feature variance indicate near-constant representations; this is a diagnostic, not proof of the cause.',
        'limitations':'No test-driven retraining or hyperparameter selection was performed.'}
(args.run/'diagnostics.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
