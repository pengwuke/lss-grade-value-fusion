"""Reference inference definitions from the frozen P4.1-C producer.
Net matches the SR1 constructor AST. Default pretrained changes True to False
to avoid downloads before strict state loading. The producer score floor is retained.
"""
import torch
from torch import nn
import timm
K = 4
RECIPE = {"drop": 0.3}

class Net(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone=timm.create_model(
            "resnet18",pretrained=pretrained,num_classes=0,global_pool="avg",in_chans=1
        )
        self.drop=nn.Dropout(RECIPE["drop"])
        self.head=nn.Linear(self.backbone.num_features,K-1)
    def forward(self,x):
        return self.head(self.drop(self.backbone(x)))

def corn_probs(logits):
    p=torch.sigmoid(logits)
    cum=torch.cumprod(p,dim=1)
    out=torch.zeros(logits.shape[0],K,device=logits.device)
    out[:,0]=1-cum[:,0]
    out[:,1]=cum[:,0]-cum[:,1]
    out[:,2]=cum[:,1]-cum[:,2]
    out[:,3]=cum[:,2]
    return out.clamp_min(1e-6)
