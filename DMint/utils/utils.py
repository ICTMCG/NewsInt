from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np
from collections import defaultdict

class Recorder():
    def __init__(self, early_step, logger):
        self.max = {'metric': -1}
        self.cur = {'metric': -1}
        self.results = defaultdict(dict)
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step
        self.logger = logger

    def add(self, x):
        self.cur = x['intent']
        self.curindex += 1
        self.logger.info(f"current[epoch{self.curindex}] (intent f1) = {self.cur['metric']}, while former max = {self.max['metric']} in epoch[{self.maxindex}]")
        judge = self.judge()
        if judge == 'save':
            self.results = x
        return judge
    
    def judge(self):
        if self.cur['metric'] > self.max['metric']:
            self.logger.info(f'From {self.max["metric"]} (intent f1 in epoch {self.maxindex}) update to {self.cur["metric"]} (intent f1 in epoch {self.curindex})')
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        if self.curindex - self.maxindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'
        
    def showfinal(self):
        self.logger.info(f"--- Epoch[{self.curindex}] max(intent f1)={self.max['metric']} ---")


def metrics(y_true, y_pred, task):
    all_metrics = {}
    try:
        all_metrics['auc'] = roc_auc_score(y_true, y_pred, average='macro')
    except:
        all_metrics['auc'] = 0
    y_pred = np.around(np.array(y_pred)).astype(int)
    all_metrics['metric'] = f1_score(y_true, y_pred, average='macro')
    all_metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro')
    all_metrics['acc'] = accuracy_score(y_true, y_pred)
    all_metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro')

    return all_metrics

def data2gpu(batch, use_cuda):
    if use_cuda:
        batch_data = {
            'idx': batch[0].cuda(),
            'content': batch[1].cuda(),
            'content_masks': batch[2].cuda(),
            'intent': batch[3].cuda(),
            'belief': batch[4].cuda(),
            'desire': batch[5].cuda(),
            'plan': batch[6].cuda()
            }
    else:
        batch_data = {
            'idx': batch[0],
            'content': batch[1],
            'content_masks': batch[2],
            'intent': batch[3],
            'belief': batch[4],
            'desire': batch[5],
            'plan': batch[6]
            }
    return batch_data

class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v