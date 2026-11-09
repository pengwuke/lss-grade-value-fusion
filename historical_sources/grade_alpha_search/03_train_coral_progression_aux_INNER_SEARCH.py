#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from common_step04a import *
import argparse
import ornet_progression_aux as prog

@torch.no_grad()
def evaluate(model, loader, device, s54):
    model.eval()
    total=0; ordinal_sum=0.0; ys=[]; probs=[]
    for x,y,r,idx in loader:
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        logits,_=model(x)
        data_loss=s54.ordinal_loss(logits,y,"coral")
        val_obj=data_loss + CNN_FC_L2*s54.ordinal_regularization(model)
        n=y.numel(); total+=n
        ordinal_sum += float(val_obj.detach().cpu())*n
        _,p=s54.cumulative_and_class_probabilities(logits,"coral")
        ys.append(y.cpu().numpy()); probs.append(p.cpu().numpy())
    yy=np.concatenate(ys); pp=np.concatenate(probs)
    return ordinal_sum/max(total,1),yy,pp,pp.argmax(1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--outer-fold",type=int,choices=range(5),required=True)
    outer=int(ap.parse_args().outer_fold)
    if not (CTRL/"00_PREFLIGHT_PASS.json").is_file(): raise RuntimeError("NO_PREFLIGHT_PASS")

    out=SCI/"coral_msaux"/f"outer_{outer}"
    if (out/"PASS_FOLD.json").is_file():
        print(f"PASS_FOLD_ALREADY_EXISTS branch=CORAL-MSaux outer={outer}"); return
    archive_partial(out)

    set_all_seeds(SEED)
    s54=load_module(SOURCE54,f"step04a_s54_coral_{outer}")
    fw=load_module(FORMAL_WRAPPER,f"step04a_fw_coral_{outer}")
    df=pd.read_csv(CTRL/"02_RESOLVED_EXPERIMENT_MANIFEST.csv",encoding="utf-8-sig")
    tr,va,te,audit=fw.role_audit(df,outer)
    vf=int(audit["fixed_validation_fold"])
    if vf!=EXPECTED_VAL[outer] or len(te)!=EXPECTED_TEST[outer]: raise RuntimeError("FOLD_ROLE_GATE_FAILED")

    train_ds=SelectedPreprocDataset(tr,"coral_msaux",True,s54,SEED)
    val_ds=SelectedPreprocDataset(va,"coral_msaux",False,s54,SEED)
    test_ds=SelectedPreprocDataset(te,"coral_msaux",False,s54,SEED)

    labels=tr[label_col(tr)].astype(int).to_numpy()
    counts=np.bincount(labels,minlength=4)
    cw=np.divide(1.0,counts,out=np.zeros_like(counts,dtype=float),where=counts>0)
    sw=torch.as_tensor(cw[labels],dtype=torch.double)
    gen=torch.Generator(); gen.manual_seed(SEED)
    sampler=WeightedRandomSampler(sw,num_samples=len(sw),replacement=True,generator=gen)

    pin=torch.cuda.is_available()
    train_loader=DataLoader(train_ds,batch_size=CNN_BATCH,shuffle=False,sampler=sampler,num_workers=0,pin_memory=pin)
    val_loader=DataLoader(val_ds,batch_size=CNN_BATCH,shuffle=False,num_workers=0,pin_memory=pin)
    test_loader=DataLoader(test_ds,batch_size=CNN_BATCH,shuffle=False,num_workers=0,pin_memory=pin)

    if not torch.cuda.is_available(): raise RuntimeError("CUDA_REQUIRED")
    device=torch.device("cuda")
    model=s54.RiskAwareOrdinalCNN("coral",dropout=CNN_DROPOUT).to(device)
    model.progression_aux=prog.ORNetProgressionAux(4).to(next(model.parameters()).device)
    for _p in model.risk_head.parameters(): _p.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=CNN_LR,weight_decay=CNN_WD)
    riskcrit=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0,device=device))

    best=float("inf"); best_epoch=None
    ck=out/"best_checkpoint_min_validation_ordinal_objective.pth"
    history=[]

    dump_json(out/"00_PROTOCOL.json",{
        "branch":"CORAL-MSaux","outer_fold":outer,"fixed_validation_fold":vf,
        "train_rows":len(tr),"val_rows":len(va),"test_rows":len(te),
        "source54":str(SOURCE54),"source54_sha256":SHA54,
        "method":"coral","msaux_training_only":True,"msaux_weight":MSAUX_WEIGHT,
        "epochs":CNN_EPOCHS,"batch_size":CNN_BATCH,"lr":CNN_LR,"weight_decay":CNN_WD,
        "fc_l2":CNN_FC_L2,"dropout":CNN_DROPOUT,
        "sampler":"inverse-frequency WeightedRandomSampler replacement=True generator seed42",
        "checkpoint_rule":"min fixed-val CORAL ordinal objective including ordinal regularization, excluding MSaux head/loss",
        "preprocessing":"DICOM-derived IQ8 + NoHE + native source54 augmentation + G1 to 64 + 0.5/0.5 normalization",
        "num_workers":0,"outer_test_used_for_training_or_checkpoint_selection":False,
    })

    print(f"CORAL_MSAUX_TRAIN_START outer={outer} train={len(tr)} val={len(va)} test={len(te)}")
    for epoch in range(1,CNN_EPOCHS+1):
        model.train(); joint_sum=ord_sum=risk_sum=0.0; seen=0
        for x,y,r,idx in train_loader:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); r=r.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits,risk_logits=model(x)
            ord_data=s54.ordinal_loss(logits,y,"coral")
            risk_value=prog.ORNetStyleFusionLoss().regloss(model.progression_aux(logits),prog.progression_targets(y).type_as(risk_logits))
            joint=(1.0-prog.SOURCE_ALPHA)*ord_data + prog.SOURCE_ALPHA*risk_value + CNN_FC_L2*(s54.ordinal_regularization(model)+model.progression_aux.reg.weight.square().sum())
            joint.backward(); opt.step()
            n=y.numel(); seen+=n
            joint_sum+=float(joint.detach().cpu())*n
            ord_sum+=float(ord_data.detach().cpu())*n
            risk_sum+=float(risk_value.detach().cpu())*n

        vloss,vy,vprob,vpred=evaluate(model,val_loader,device,s54)
        vm=metrics_from_arrays(vy,vpred)
        history.append({
            "epoch":epoch,"train_joint_loss":joint_sum/seen,
            "train_ordinal_data_loss":ord_sum/seen,"train_risk_loss":risk_sum/seen,
            "val_ordinal_objective":vloss,
            **{f"val_{k}":v for k,v in vm.items()}
        })
        pd.DataFrame(history).to_csv(out/"training_history.csv",index=False,encoding="utf-8-sig")

        if vloss<best:
            best=vloss; best_epoch=epoch
            torch.save({
                "model_state_dict":copy.deepcopy(model.state_dict()),
                "best_epoch":epoch,"best_validation_ordinal_objective":best,
                "outer_fold":outer,"fixed_validation_fold":vf,
            },ck)

        if epoch==1 or epoch%10==0 or epoch==CNN_EPOCHS:
            print(
                f"[CORAL-MSaux outer={outer}] epoch={epoch:03d}/{CNN_EPOCHS} "
                f"train_joint={joint_sum/seen:.6f} val_ord={vloss:.6f} "
                f"val_qwk={vm['qwk']:.4f} val_macro_f1={vm['macro_f1']:.4f} "
                f"best_epoch={best_epoch}",flush=True
            )

    dump_json(out/"checkpoint_freeze_before_outer_test.json",{
        "outer_fold":outer,"fixed_validation_fold":vf,"best_epoch":best_epoch,
        "best_validation_ordinal_objective":best,"checkpoint":str(ck),
        "checkpoint_sha256":sha256(ck),"outer_test_opened":False,
    })

    obj=torch_load(ck,"cpu")
    model.load_state_dict(obj["model_state_dict"],strict=True); model=model.to(device)
    print('INNER_SEARCH_STOP_BEFORE_OUTER_TEST=PASS',flush=True)
    return
    tloss,ty,tprob,tpred=evaluate(model,test_loader,device,s54)

    pred=te.reset_index(drop=True).copy()
    pred["y_true"]=ty; pred["y_pred"]=tpred
    for k in range(4): pred[f"prob_{k}"]=tprob[:,k]
    pred["outer_fold_protocol"]=outer; pred["fixed_validation_fold_protocol"]=vf
    pred.to_csv(out/"outer_test_predictions.csv",index=False,encoding="utf-8-sig")

    tm=metrics_from_arrays(ty,tpred)
    dump_json(out/"PASS_FOLD.json",{
        "status":"PASS_STEP04A_CORAL_MSAUX_FOLD","outer_fold":outer,
        "fixed_validation_fold":vf,"rows":len(pred),"best_epoch":best_epoch,
        "checkpoint_sha256":sha256(ck),"outer_test_metrics":tm,
        "outer_test_used_for_training_or_checkpoint_selection":False,
    })
    print(f"PASS_STEP04A_CORAL_MSAUX_FOLD outer={outer} rows={len(pred)} best_epoch={best_epoch} qwk={tm['qwk']:.6f} acc={tm['accuracy']:.6f}")

if __name__=="__main__":
    try: main()
    except Exception as e:
        print(f"FATAL_CORAL_MSAUX_FOLD | {type(e).__name__}: {e}",file=sys.stderr)
        raise
