import logging
import os
import json
import torch
import random
import numpy as np
from collections import defaultdict

from models.dmint import Trainer as DMINTTrainer

class Run():
    def __init__(self, config):
        self.config = config
        self._init_param()
        self._init_seed()
        self.logger = self._init_logger()
    
    def _init_param(self):
        self.config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config['save_param_dir'] = os.path.join(self.config['save_param_dir'], self.config['model_name'], self.config['custom_name'])
        os.makedirs(self.config['save_param_dir'], exist_ok=True)
        print(f"Save log in: {self.config['save_param_dir']}")
        if len(self.config['infer_save']) > 0:
            os.makedirs(self.config['infer_save'], exist_ok=True)
               
    def _init_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(os.path.join(self.config['save_param_dir'] + f"/{self.config['custom_name']}.log"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger
    
    def _init_seed(self):
        seed = self.config['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def main(self):
        if self.config['eval']:
            print('eval mode, pretrained_model: {}'.format(self.config['checkpoint']))
            trainer = DMINTTrainer(self.config)
            trainer.train(self.logger)
            return
        train_params = {
            'lr': [2e-5],
        }
        print(train_params)
        best_param = []
        json_result = defaultdict(list)

        for p, vs in train_params.items():
            best_metric = defaultdict(dict)
            best_metric['intent']['metric'] = 0
            best_v = vs[0]
            best_model_path = None
            for v in vs:
                if p == 'lr':
                    self.config['lr'] = v
                
                if self.config['model_name'] == 'dmint':
                    trainer = DMINTTrainer(self.config)
                elif self.config['model_name'] == 'textcnn':
                    trainer = TextCNNTrainer(self.config)
                metrics, model_path = trainer.train(self.logger)
                json_result[p].append(metrics)
                if metrics['intent']['metric'] > best_metric['intent']['metric']:
                    best_metric = metrics
                    best_v = v
                    best_model_path = model_path
            best_param.append({p: best_v})
            print(f"=== Finish grid search for [{p}] ===")

            self.logger.info(f"=== Finish grid search for [{p}] ===")
            self.logger.info(f"best model path: {best_model_path}")
            self.logger.info(f"best param [{p}]: {best_v}")
            self.logger.info(f"best metric: {best_metric}")
        
        metrics = trainer.test(self.logger)

        with open(os.path.join(self.config['save_param_dir'], 'best_results.json'), 'w') as f:
            json.dump(json_result, f, indent=4, ensure_ascii=False)
        with open(os.path.join(self.config['save_param_dir'], 'best_param.json'), 'w') as f:
            json.dump(best_param, f, indent=4, ensure_ascii=False)
        self.logger.info(f"Finished all. Saving best parameters in {self.config['save_param_dir']}.")
        print(f"Finished all. Logger and results saved in {self.config['save_param_dir']}/.")
        return