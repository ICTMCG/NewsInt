import os
import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default="DMINT")
parser.add_argument("--batchsize", type=int, default=64)
# === Load & Save ===
parser.add_argument("--data_path", type=str, default="data/")
parser.add_argument('--save_log_dir', default= './logs')
parser.add_argument('--save_param_dir', default= './param_model')
parser.add_argument('--custom_name', default= 'demo')
parser.add_argument("--log_step", type=int, default=100)
parser.add_argument("--save_step", type=int, default=100)
parser.add_argument("--save_output", action='store_true')
parser.add_argument("--explanation", action='store_true')
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--test_type', default='merge', type=str)
parser.add_argument('--train_type', default='select', type=str)

# === Train ===
parser.add_argument('--clip_max_norm', type=float, default=0.1)
parser.add_argument('--early_stop', type=int, default=5)
parser.add_argument("--eval", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--seed", type=int, default=2023)
parser.add_argument("--gpu", type=int, default=1)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument('--emb_dim', type=int, default=768)
parser.add_argument('--emb_type', default='bert')
parser.add_argument("--max_len", type=int, default=512)
parser.add_argument("--belief_expert_layer_num", type=int, default=1)
parser.add_argument("--desire_expert_layer_num", type=int, default=1)
parser.add_argument("--plan_expert_layer_num", type=int, default=1)
parser.add_argument("--backbone", default='bert')
parser.add_argument("--gru_layers", type=int, default=1)
parser.add_argument("--layer11", action="store_true")

# === Infer ===
parser.add_argument('--checkpoint', type=str, default="")
parser.add_argument('--eval_customer_dataset_path', type=str, default="")
parser.add_argument('--infer_task_path', type=str, default="")
parser.add_argument('--file_name', type=str, default="news_to_infer.json")

# === Hyperparameters ===
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--cls_dims", type=int, default=128)
parser.add_argument("--loss_weight_belief", type=int, default=1)
parser.add_argument("--loss_weight_desire", type=int, default=1)
parser.add_argument("--loss_weight_plan", type=int, default=1)
parser.add_argument("--loss_weight_intent", type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=5e-5)

# === Ablation ===
parser.add_argument("--module_ablation", action="store_true")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

from grid_search import Run

config = {
    'model_name': args.model_name,
    'batchsize': args.batchsize,
    'cls_num_plan': 2,
    'cls_num_belief': 3,
    'cls_num_desire': 4,
    'cls_num_intent': 2,
    
    'data_path': args.data_path,
    'save_log_dir': args.save_log_dir,
    'save_param_dir': args.save_param_dir,
    'custom_name': args.custom_name,
    'log_step': args.log_step,
    'save_step': args.save_step,
    'save_output': args.save_output,
    'explanation': args.explanation,
    'num_workers': args.num_workers,
    'test_type': args.test_type,
    'train_type': args.train_type,

    'clip_max_norm': args.clip_max_norm,
    'early_stop': args.early_stop,
    'eval': args.eval,
    'debug': args.debug,
    'seed': args.seed,
    'gpu': args.gpu,
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'emb_dim': args.emb_dim,
    'emb_type': args.emb_type,
    'max_len': args.max_len,
    'belief_expert_layer_num': args.belief_expert_layer_num,
    'desire_expert_layer_num': args.desire_expert_layer_num,
    'plan_expert_layer_num': args.plan_expert_layer_num,
    'backbone': args.backbone,
    'gru_layers': args.gru_layers,
    'layer11': args.layer11,

    'checkpoint': args.checkpoint,
    'eval_customer_dataset_path': args.eval_customer_dataset_path,
    'infer_task_path': args.infer_task_path,
    'file_name': args.file_name,
    'infer_save': os.path.join(args.eval_customer_dataset_path, args.infer_task_path, 'dmint_infer/'),

    'dropout': args.dropout,
    'lr': args.lr,
    'model':
            {
            'mlp': {'dims': [128, 64], 'dropout': 0.2}
            },
    'loss_weight_intent': args.loss_weight_intent,
    'loss_weight_belief': args.loss_weight_belief,
    'loss_weight_desire': args.loss_weight_desire,
    'loss_weight_plan': args.loss_weight_plan,
    'weight_decay': args.weight_decay,
}

if __name__ == "__main__":
    if args.eval:
        print('Eval, checkpoint: {}'.format(args.checkpoint))
    else:
        print(f'Train, model name: {args.model_name}; lr: {args.lr}; gpu: {args.gpu}; batchsize: {args.batchsize}; \
              epoch: {args.epochs}; loss_weight_belief: {args.loss_weight_belief}; loss_weight_desire: {args.loss_weight_desire}; loss_weight_plan: {args.loss_weight_plan}')
        
    Run(config=config).main()
    