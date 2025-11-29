import os
import torch
import torch.nn as nn
from sklearn.metrics import *
from transformers import RobertaModel
from utils.utils import Recorder
from utils.dataset import Dataset_Creator, CustomerDataset
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from engine import train_one_epoch, evaluate
import shutil
import datetime
import time
import copy
import csv

class ExtractorLocal(nn.Module):
    def __init__(self, input_dim, feature_kernel, dropout):
        super(ExtractorLocal, self).__init__()
        # self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, padding=kernel_size // 2)
        self.convs = nn.ModuleList(
            [nn.Conv1d(input_dim, output_dim, kernel_size) for kernel_size, output_dim in feature_kernel.items()]
        )
        self.dropout = nn.Dropout(dropout)
        # self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = [conv(x) for conv in self.convs]
        x = [torch.max_pool1d(f, f.shape[-1]) for f in x]
        x = torch.cat(x, dim=1)
        x = x.view([-1, x.shape[1]])
        x = self.dropout(x)
        # x = self.relu(x)
        return x

class MLP(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, output_dim, dropout):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            input_dim = embed_dim
        if output_dim > 0:
            layers.append(torch.nn.Linear(input_dim, output_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)
        return x
    
class MaskAttention(nn.Module):
    def __init__(self, input_shape):
        super(MaskAttention, self).__init__()
        self.attention_layer = torch.nn.Linear(input_shape, 1)

    def forward(self, inputs, mask=None):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        if mask is not None:
            scores = scores.masked_fill(mask==0, float("-inf"))
        scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)

        return outputs, scores


class DMINT(torch.nn.Module):
    def __init__(self,
                 hidden_dim=768,
                 config=None):
        super().__init__()
        self.config = config
        self.bert = RobertaModel.from_pretrained(os.path.expanduser('~/utils/roberta-base-model')).requires_grad_(False)
        if config['layer11']:
            for name, param in self.bert.named_parameters():
                if name.startswith("encoder.layer.11"): \
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        self.embedding = self.bert.embeddings
        expert_num = 0

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        rnn_output_shape = sum([feature_kernel[kernel] for kernel in feature_kernel]) # 320

        if config['belief_expert_layer_num'] > 0:
            self.belief_expert = ExtractorLocal(hidden_dim, feature_kernel, config['dropout'])
            self.belief_embedding = nn.Embedding(1, hidden_dim)
            expert_num += 1
        else:
            self.belief_expert = None
            self.belief_embedding = None
        self.belief_classifier = MLP(len(feature_kernel) * 64, config['model']['mlp']['dims'], config['cls_num_belief'], config['model']['mlp']['dropout'])
        self.belief_classifier_base = MLP(config['emb_dim'], config['model']['mlp']['dims'], config['cls_num_belief'], config['model']['mlp']['dropout'])

        if config['desire_expert_layer_num'] > 0:
            self.desire_expert = ExtractorLocal(hidden_dim, feature_kernel, config['dropout'])
            self.desire_embedding = nn.Embedding(1, hidden_dim)
            expert_num += 1
        else:
            self.desire_expert = None
            self.desire_embedding = None
        self.desire_classifier = MLP(len(feature_kernel) * 64, config['model']['mlp']['dims'], config['cls_num_desire'], config['model']['mlp']['dropout'])
        self.desire_classifier_base = MLP(config['emb_dim'], config['model']['mlp']['dims'], config['cls_num_desire'], config['model']['mlp']['dropout'])

        if config['plan_expert_layer_num'] > 0:
            self.plan_expert = ExtractorLocal(hidden_dim, feature_kernel, config['dropout'])
            self.plan_embedding = nn.Embedding(1, hidden_dim)
            expert_num += 1
        else:
            self.plan_expert = None
            self.plan_embedding = None
        self.plan_classifier = MLP(len(feature_kernel) * 64, config['model']['mlp']['dims'], config['cls_num_plan'], config['model']['mlp']['dropout'])
        self.plan_classifier_base = MLP(config['emb_dim'], config['model']['mlp']['dims'], config['cls_num_plan'], config['model']['mlp']['dropout'])

        self.attention = MaskAttention(hidden_dim)
        self.intent_classifier = MLP(input_dim=len(feature_kernel) * 64, embed_dims=config['model']['mlp']['dims'], output_dim=config['cls_num_intent'], dropout=config['model']['mlp']['dropout'])
        self.intent_classifier_base = MLP(input_dim=config['emb_dim'], embed_dims=config['model']['mlp']['dims'], output_dim=config['cls_num_intent'], dropout=config['model']['mlp']['dropout'])

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim*2, 384),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(384, expert_num),
            nn.Softmax(dim=1)
        )

        self.criterion_weight_dict = {
            "loss_intent": config['loss_weight_intent'],
            "loss_belief": config['loss_weight_belief'],
            "loss_desire": config['loss_weight_desire'],
            "loss_plan": config['loss_weight_plan'],
        }

        self.ExtractorGlobal = nn.ModuleList([copy.deepcopy(self.bert.encoder.layer[-1]) for _ in range(expert_num)])
        for param in self.ExtractorGlobal.parameters():
            param.requires_grad = True
    
    def forward(self, content, content_masks):
        news_tokens = self.bert(content, attention_mask=content_masks)[0]
        gate_input_feature, _ = self.attention(news_tokens, content_masks) # [bs, 768]

        extended_attention_mask = content_masks[:, None, None, :] # [64, 1, 1, 512]
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(content_masks.dtype).min
        # extended_attention_mask = (~extended_attention_mask)

        # === Experts ===
        gate_idx = 0
        expert_embedding = []
        if self.belief_expert is not None:
            belief_feature = self.ExtractorGlobal[gate_idx](news_tokens, extended_attention_mask)[0]
            belief_feature = self.belief_expert(belief_feature)
            belief_cls_out = torch.sigmoid(self.belief_classifier(belief_feature))
            expert_embedding.append(self.belief_embedding.weight)
            gate_idx += 1
        else:
            belief_feature = belief_cls_out = None
        if self.desire_expert is not None:
            desire_feature = self.ExtractorGlobal[gate_idx](news_tokens, extended_attention_mask)[0]
            desire_feature = self.desire_expert(desire_feature)
            desire_cls_out = torch.sigmoid(self.desire_classifier(desire_feature))
            expert_embedding.append(self.desire_embedding.weight)
            gate_idx += 1
        else:
            desire_feature = desire_cls_out = None
        if self.plan_expert is not None:
            plan_feature = self.ExtractorGlobal[gate_idx](news_tokens, extended_attention_mask)[0]
            plan_feature = self.plan_expert(plan_feature)
            plan_cls_out = torch.sigmoid(self.plan_classifier(plan_feature))
            expert_embedding.append(self.plan_embedding.weight)
            gate_idx += 1
        else:
            plan_feature = plan_cls_out = None
        
        # === Gate ===
        if expert_embedding:
            shared_feature = 0
            gate_idx = 0
            expert_embedding = torch.cat(expert_embedding, dim=0).mean(dim=0, keepdim=True).expand(gate_input_feature.size(0), -1)
            gate_value = self.gate(torch.cat([gate_input_feature, expert_embedding], dim=-1)) # [0.9300, 0.0636, 0.0064]
            for feature in [belief_feature, desire_feature, plan_feature]:
                if feature is not None:
                    shared_feature += feature * gate_value[:, gate_idx].unsqueeze(1)
                    gate_idx += 1
            intent_cls = self.intent_classifier(shared_feature)
        else:
            shared_feature = gate_input_feature
            intent_cls = self.intent_classifier_base(shared_feature)
            gate_value = None

        intent_cls_out = torch.sigmoid(intent_cls)
        if belief_cls_out is None:
            belief_cls_out = torch.sigmoid(self.belief_classifier_base(shared_feature))
        if desire_cls_out is None:
            desire_cls_out = torch.sigmoid(self.desire_classifier_base(shared_feature))
        if plan_cls_out is None:
            plan_cls_out = torch.sigmoid(self.plan_classifier_base(shared_feature))
        if self.config['save_output']:
            return intent_cls_out, belief_cls_out, desire_cls_out, plan_cls_out, shared_feature.squeeze(), belief_feature.squeeze(), desire_feature.squeeze(), plan_feature.squeeze()
        elif self.config['explanation'] and self.config['eval']:
            return gate_value, intent_cls_out, belief_cls_out, desire_cls_out, plan_cls_out
        else:
            return intent_cls_out, belief_cls_out, desire_cls_out, plan_cls_out


class Trainer():
    def __init__(self, 
                 config
                 ):
        self.config = config

    def train(self, logger = None):
        if(logger):
            logger.info('Start training......')
            logger.info(f"lr: {self.config['lr']}, save_path: {self.config['save_param_dir']}")
        self.model = DMINT(config=self.config).to(self.config['device'])

        # === Dataset ===
        if len(self.config['eval_customer_dataset_path']) > 0:
            infer_data_path = os.path.join(self.config['eval_customer_dataset_path'], self.config['infer_task_path'])
            dataset_val = CustomerDataset(dataset_path=infer_data_path, file_name=self.config['file_name'], max_len=self.config['max_len'])
        else:
            dataset_creator = Dataset_Creator(config = self.config)
            dataset_val = dataset_creator.build_dataset("val")
        sampler_val = SequentialSampler(dataset_val)
        data_loader_val = DataLoader(dataset_val, self.config['batchsize'], sampler=sampler_val, drop_last=False, num_workers=self.config['num_workers'])
        
        if not self.config['eval']:
            dataset_train = dataset_creator.build_dataset("train")
            sampler_train = RandomSampler(dataset_train)
            batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, self.config['batchsize'], drop_last=True)
            data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=self.config['num_workers'])

        # === Eval ===
        if self.config['eval']:
            checkpoint = torch.load(self.config['checkpoint'], map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
            results = evaluate(self.model, data_loader_val, self.config['device'], config=self.config)
            results_save_csv = os.path.join(self.config['save_param_dir'], f"{self.config['custom_name']}_{self.config['lr']}_results.csv")
            
            with open(results_save_csv, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['metric', 'intent', 'belief', 'desire', 'plan'])
                writer.writerow(['auc', results['intent']['auc'], results['belief']['auc'], results['desire']['auc'], results['plan']['auc']])
                writer.writerow(['macro_f1', results['intent']['metric'], results['belief']['metric'], results['desire']['metric'], results['plan']['metric']])
                writer.writerow(['macro_precision', results['intent']['macro_precision'], results['belief']['macro_precision'], results['desire']['macro_precision'], results['plan']['macro_precision']])
                writer.writerow(['acc', results['intent']['acc'], results['belief']['acc'], results['desire']['acc'], results['plan']['acc']])
                writer.writerow(['macro_recall', results['intent']['macro_recall'], results['belief']['macro_recall'], results['desire']['macro_recall'], results['plan']['macro_recall']])
            logger.info(f"Save infer results to {results_save_csv}")
            print(f"Save infer results to {results_save_csv}")
            logger.info('Finish inference!')
            if self.config['save_output']:
                logger.info(f"Save infer to {self.config['infer_save']}")
                print(f"Save infer to {self.config['infer_save']}")
            logger.info(results)
            return
        
        # === Train ===
        optimizer = torch.optim.AdamW([{"params": [p for n, p in self.model.named_parameters() if p.requires_grad and "extra_bert" not in n]},
                                   {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and "extra_bert" in n]}], 
                                   lr=self.config['lr'], 
                                   weight_decay=self.config['weight_decay'])
        recorder = Recorder(self.config['early_stop'], logger)
        shutil.copy("models/mint.py", self.config['save_param_dir'])
        shutil.copy("./run.sh", self.config['save_param_dir'])
        for epoch in range(self.config['epochs']):
            epoch_start_time = time.time()
            train_one_epoch(self.model, data_loader_train, optimizer, self.config['device'], epoch, self.config['clip_max_norm'], config=self.config)
            epoch_time = time.time() - epoch_start_time
            epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
            print('Epoch training time {}'.format(epoch_time_str))
                        
            results = evaluate(self.model, data_loader_val, self.config['device'], config=self.config)
            mark = recorder.add(results)
            if mark == 'save':
                checkpoint_paths = os.path.join(self.config['save_param_dir'], 'checkpoint.pth')
                weights = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': self.config,
                }
                torch.save(weights, checkpoint_paths)
                logger.info(f"Save model to {checkpoint_paths}")
                print(f"Save model to {checkpoint_paths}")
                results_save_csv = os.path.join(self.config['save_param_dir'], 
                                                f"{self.config['lr']}_{self.config['loss_weight_belief']}_{self.config['loss_weight_desire']}_{self.config['loss_weight_plan']}_{self.config['loss_weight_intent']}_results.csv")
                with open(results_save_csv, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['metric', 'intent', 'belief', 'desire', 'plan'])
                    writer.writerow(['auc', results['intent']['auc'], results['belief']['auc'], results['desire']['auc'], results['plan']['auc']])
                    writer.writerow(['macro_f1', results['intent']['metric'], results['belief']['metric'], results['desire']['metric'], results['plan']['metric']])
                    writer.writerow(['macro_precision', results['intent']['macro_precision'], results['belief']['macro_precision'], results['desire']['macro_precision'], results['plan']['macro_precision']])
                    writer.writerow(['acc', results['intent']['acc'], results['belief']['acc'], results['desire']['acc'], results['plan']['acc']])
                    writer.writerow(['macro_recall', results['intent']['macro_recall'], results['belief']['macro_recall'], results['desire']['macro_recall'], results['plan']['macro_recall']])
                logger.info(f"Save infer results to {results_save_csv}")
                print(f"Save infer results to {results_save_csv}")
            elif mark == 'esc':
                break
            else:
                continue
        logger.info("=== Finish one run ===")
        logger.info(f"{recorder.results}")
        logger.info("======================")
        print(f"=== Finish one run ===")

        return recorder.results, self.config['save_param_dir']


    def test(self, logger=None):
        if logger:
            logger.info("Start testing......")

        self.model = DMINT(config=self.config).to(self.config['device'])

        checkpoint_path = os.path.join(self.config['save_param_dir'], 'checkpoint.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Test based on checkpoint saved in {checkpoint_path}.")
        self.model.load_state_dict(checkpoint['model'])
        if logger:
            logger.info(f"Loaded model from {checkpoint_path}")

        dataset_creator = Dataset_Creator(config=self.config)
        dataset_test = dataset_creator.build_dataset("test")
        sampler_test = SequentialSampler(dataset_test)
        sampler_test = SequentialSampler(dataset_test)
        data_loader_test = DataLoader(
            dataset_test,
            self.config['batchsize'],
            sampler=sampler_test,
            drop_last=False,
            num_workers=self.config['num_workers']
        )

        results = evaluate(self.model, data_loader_test, self.config['device'], config=self.config)

        results_save_csv = os.path.join(
            self.config['save_param_dir'],
            f"{self.config['custom_name']}_{self.config['lr']}_test_results.csv"
        )

        with open(results_save_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'intent', 'belief', 'desire', 'plan'])
            writer.writerow(['auc', results['intent']['auc'], results['belief']['auc'], results['desire']['auc'], results['plan']['auc']])
            writer.writerow(['macro_f1', results['intent']['metric'], results['belief']['metric'], results['desire']['metric'], results['plan']['metric']])
            writer.writerow(['macro_precision', results['intent']['macro_precision'], results['belief']['macro_precision'], results['desire']['macro_precision'], results['plan']['macro_precision']])
            writer.writerow(['acc', results['intent']['acc'], results['belief']['acc'], results['desire']['acc'], results['plan']['acc']])
            writer.writerow(['macro_recall', results['intent']['macro_recall'], results['belief']['macro_recall'], results['desire']['macro_recall'], results['plan']['macro_recall']])
        
        if logger:
            logger.info(f"Save test results to {results_save_csv}")
            logger.info("Finish testing!")
            logger.info(results)
        
        print(f"Save test results to {results_save_csv}")
        print("=== Testing finished ===")

        return results
