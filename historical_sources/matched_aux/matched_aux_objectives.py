"""Four matched training objectives derived from the SHA-locked R09 trainer.
No data loading, file writes, model selection, or new prediction decoder here.
"""
import torch
import torch.nn as nn

ARM_NAMES = ('A0_NATURAL', 'A1_SCALED', 'A2_BINARY_P', 'A3_GRADE_FIXED')
ALPHA = 0.125


def configure_arm(model, arm):
    if arm not in ARM_NAMES:
        raise ValueError('UNKNOWN_ABLATION_ARM')
    # Every arm has already instantiated both heads: never perturb the initial
    # shared-network RNG stream by removing a head before initialization.
    for p in model.risk_head.parameters():
        p.requires_grad_(False)
    for p in model.progression_aux.parameters():
        p.requires_grad_(arm in ('A2_BINARY_P', 'A3_GRADE_FIXED'))


def objective(model, logits, risk_logits, y, arm, s54, prog, fc_l2):
    if arm not in ARM_NAMES:
        raise ValueError('UNKNOWN_ABLATION_ARM')
    ord_data = s54.ordinal_loss(logits, y, 'coral')
    if arm == 'A0_NATURAL':
        joint = ord_data + fc_l2 * s54.ordinal_regularization(model)
        aux = ord_data.new_zeros(())
    elif arm == 'A1_SCALED':
        joint = (1.0-ALPHA) * ord_data + fc_l2 * s54.ordinal_regularization(model)
        aux = ord_data.new_zeros(())
    else:
        score = model.progression_aux(logits)
        if arm == 'A2_BINARY_P':
            aux = nn.BCEWithLogitsLoss()(score, (y >= 2).type_as(risk_logits))
        else:
            # Exact mathematical expression and evaluation order of R09 alpha=.125.
            aux = prog.ORNetStyleFusionLoss().regloss(
                score, prog.progression_targets(y).type_as(risk_logits))
        joint = (1.0-ALPHA)*ord_data + ALPHA*aux + fc_l2*(
            s54.ordinal_regularization(model) + model.progression_aux.reg.weight.square().sum())
    return joint, ord_data, aux
