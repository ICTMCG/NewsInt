import json
import torch
import torch.nn.functional as F
from typing import Iterable
from collections import defaultdict
import math
import sys
import os
from utils.utils import data2gpu, Averager, metrics
import utils.misc as utils

def train_one_epoch(model: torch.nn.Module,
                    data_loder: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    max_norm: float = 1.0,
                    config=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch+1)
    print_freq = config['log_step']
    avg_loss = Averager()

    for samples in metric_logger.log_every(data_loder, print_freq, header):
        idxs, content_token_ids, content_masks, intents, beliefs, desires, plans = [sample.to(device) for sample in samples]
        outputs = model(content_token_ids, content_masks)
        
        loss_dict = get_criterion(outputs, [intents, beliefs, desires, plans])
        weight_dict = model.criterion_weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(losses):
            print("Loss is {} (isfinite), stopping training".format(losses))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        avg_loss.add(losses.item())

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)


@torch.no_grad()
def evaluate(model, data_loader, device, config=None):
    model.eval()
    task_name = ['intent', 'belief', 'desire', 'plan']
    label = defaultdict(list)
    pred = defaultdict(list)
    gate = []
    idx_list = []
    eva = defaultdict()

    metric_logger = utils.MetricLogger(delimiter="  ")
    print_freq = config['log_step']
    header = 'Test:'
    for samples in metric_logger.log_every(data_loader, print_freq, header):
        with torch.no_grad():
            if len(config['eval_customer_dataset_path']) > 0 and config['save_output']:
                news_ids, content_token_ids, content_masks = [sample.to(device) 
                                                              if isinstance(sample, torch.Tensor) else sample 
                                                              for sample in samples]
                outputs_deteached = []
                outputs = model(content_token_ids, content_masks)
                for output in outputs:
                    if output is not None:
                        outputs_deteached.append(output.detach().cpu())
                    else:
                        outputs_deteached.append([None] * len(news_ids))
                intent_cls_out, belief_cls_out, desire_cls_out, plan_cls_out, \
                    intent_embeddings, belief_embeddings, desire_embeddings, plan_embeddings = outputs_deteached
                belief_mapper={0:'against', 1:'favor', 2:'neither'}
                desire_mapper={0:'economic_interest', 1:'political_interest', 2:'psychological_fulfillment', 3:'public_interest'}
                plan_mapper={0:'fairness', 1:'tinted'}
                intent_mapper={0:'harmful', 1:'unharmful'}
                for [news_id, intent_cls, belief_cls, desire_cls, plan_cls, \
                     intent_embedding, belief_embedding, desire_embedding, plan_embedding] in \
                        zip(news_ids, intent_cls_out, belief_cls_out, desire_cls_out, plan_cls_out, \
                            intent_embeddings, belief_embeddings, desire_embeddings, plan_embeddings):
                    save_dic = {
                        "id": news_id,
                        "intent_cls": intent_mapper[intent_cls.tolist().index(max(intent_cls.tolist()))],
                        "belief_cls": belief_mapper[belief_cls.tolist().index(max(belief_cls.tolist()))],
                        "desire_cls": desire_mapper[desire_cls.tolist().index(max(desire_cls.tolist()))],
                        "plan_cls": plan_mapper[plan_cls.tolist().index(max(plan_cls.tolist()))], # tensor([0.9785, 0.0217])
                        "intent_embedding": intent_embedding,
                        "belief_embedding": belief_embedding,
                        "desire_embedding": desire_embedding,
                        "plan_embedding": plan_embedding
                    }
                    torch.save(save_dic, os.path.join(config['infer_save'], f"{news_id}.pt"))
            else:
                batch_data = data2gpu(samples, torch.cuda.is_available())
                batch_pred = model(batch_data['content'], batch_data['content_masks'])
                if config['explanation'] and config['eval']:
                    for i in range(len(task_name)):
                        label[task_name[i]].extend(batch_data[task_name[i]].detach().cpu().numpy().tolist())
                    for i in range(len(batch_pred)):
                        if i == 0:
                            gate.extend(batch_pred[i].detach().cpu().numpy().tolist())
                        else:
                            if batch_pred[i] is not None:
                                pred[task_name[i-1]].extend(batch_pred[i].detach().cpu().numpy().tolist())
                else:
                    for i in range(len(batch_pred)):
                        if batch_pred[i] is not None:
                            idx_list.extend(batch_data['idx'].detach().cpu().numpy().tolist())
                            pred[task_name[i]].extend(batch_pred[i].detach().cpu().numpy().tolist())
                            label[task_name[i]].extend(batch_data[task_name[i]].detach().cpu().numpy().tolist())
    if len(config['eval_customer_dataset_path']) == 0:
        pred_save_path = os.path.join(config['save_param_dir'], f"{config['custom_name']}_pred.json")
        for task_name in label.keys():
            eva[task_name] = metrics(label[task_name], pred[task_name], task=task_name)
        if config['eval']:
            pred_save = defaultdict(dict)
            for i, idx in enumerate(idx_list[:200]):
                for task_name in label.keys():
                    pred_save[idx][task_name] = pred[task_name][i]
            json.dump(pred_save, open(pred_save_path, 'w'), indent=4)
            print(f"Save prediction in: {pred_save_path}")

    if config['explanation'] and config['eval']:
        out = []
        for i in range(len(gate)):
            out_item = {}
            out_item['id'] = i
            out_item['gate'] = gate[i]
            for task_name in label.keys():
                out_item[f"{task_name}_pred"] = pred[task_name][i]
                out_item[f"{task_name}_label"] = label[task_name][i]
            out.append(out_item)
        save_path = os.path.join(config['save_param_dir'], f"{config['custom_name']}_explanation.json")
        json.dump(out, open(save_path, 'w'), indent=4)

    return eva

def get_loss():
    # return F.binary_cross_entropy
    return F.cross_entropy
    
def get_criterion(outputs, targets):
    loss_dic = {}
    for [loss, pred, tgt] in zip(["loss_intent", "loss_belief", "loss_desire", "loss_plan"], outputs, targets):
        loss_fun = get_loss()
        if pred is not None:
            loss_dic[loss] = loss_fun(pred, tgt)
        
    return loss_dic