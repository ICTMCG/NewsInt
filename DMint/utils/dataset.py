import os
import json
import torch
from transformers import RobertaTokenizer
from torch.utils.data import TensorDataset, Dataset

tokenizer = RobertaTokenizer.from_pretrained(os.path.expanduser("~/utils/roberta-base-token/"))

class Dataset_Creator:
    def __init__(self, config):
        self.data_path = config['data_path']
        self.max_len = config['max_len']
        self.batch_size = config['batchsize']
        self.num_workers = config['num_workers']
        self.config = config
        
        self.belief = set()
        self.desire = set()
        self.plan = set()
        self.intent = set()

        self.all_labels = {"train": None, "test": None}
        self.news_base = None
        self.news_topic = None

        for sub_set in ["train", "val", "test"]:
            self.load_data(sub_set)

    def build_dataset(self, sub_set):
        assert sub_set in self.all_labels.keys()
        sub_labels = self.all_labels[sub_set]
        belief_mapper = {k: v for v, k in enumerate(sorted(self.belief))}
        desire_mapper = {k: v for v, k in enumerate(sorted(self.desire))}
        plan_mapper = {k: v for v, k in enumerate(sorted(self.plan))}
        intent_mapper = {k: v for v, k in enumerate(sorted(self.intent))}
        # print(f"Check category:\nbelief: {belief_mapper}\ndesire: {desire_mapper}\nplan: {plan_mapper}\nintent: {intent_mapper}\n")
        
        idxs, contents, beliefs, desires, plans, intents = [], [], [], [], [], []
        for sub_label in sub_labels:
            idx, belief, desire, plan, intent = sub_label
            cur_news = self.news_base[idx]
            topic = self.news_topic[idx]
            cur_content = topic + '[SEP]' + cur_news['title'] + '[SEP]' + cur_news['content']
            contents.append(cur_content)
            intents.append(torch.zeros(len(self.intent)).scatter_(0, torch.tensor(intent_mapper[intent]), 1))
            beliefs.append(torch.zeros(len(self.belief)).scatter_(0, torch.tensor(belief_mapper[belief]), 1))
            desires.append(torch.zeros(len(self.desire)).scatter_(0, torch.tensor([desire_mapper[sub_desire] for sub_desire in desire]), 1))
            plans.append(torch.zeros(len(self.plan)).scatter_(0, torch.tensor(plan_mapper[plan]), 1))
            idxs.append(int(idx))
        content_token_ids, content_masks = word2input(contents, self.max_len)
        
        dataset = TensorDataset(torch.tensor(idxs),
                                content_token_ids,
                                content_masks,
                                torch.stack(intents, dim=0),
                                torch.stack(beliefs, dim=0),
                                torch.stack(desires, dim=0),
                                torch.stack(plans, dim=0))
        return dataset

    
    def load_data(self, sub_set):
        sub_set_file = sub_set+'.csv'
        sub_set_path = os.path.join(self.data_path, sub_set_file)
        sub_set_labels = self.load_label(sub_set_path)
        self.all_labels[sub_set] = sub_set_labels
        print(f"Load {sub_set_file} data: {len(sub_set_labels)}")

        if self.news_base is None:
            with open(os.path.join(self.data_path, "news_docs.json"), 'r') as f:
                self.news_base = json.load(f)
        if self.news_topic is None:
            with open(os.path.join(self.data_path, "news_topic.json"), 'r') as f:
                self.news_topic = json.load(f)

    def load_label(self, path):
        lables = []
        with open(path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                idx, belief, desire, plan, intent = [x.lower() for x in line.split(',')]
                # update global dataset for check
                self.belief.add(belief)
                desire = desire.split("&")
                for sub in desire:
                    self.desire.add(sub)
                self.plan.add(plan)
                self.intent.add(intent)
                
                lables.append([idx, belief, desire, plan, intent])
        return lables

class CustomerDataset(Dataset):
    # For model inferring (in application tasks)
    def __init__(self, dataset_path, file_name, max_len):
        self.dataset_path = dataset_path
        self.max_len = max_len
        with open(os.path.join(self.dataset_path, file_name), 'r') as f:
            self.news_base = json.load(f)
        print(f"Load {file_name} data: {len(self.news_base)}")

    def __getitem__(self, index):
        cur_news = self.news_base[index]
        newsid = cur_news['id']

        # Application sample
        if 'fnd' in self.dataset_path:
            cur_content = str(cur_news['content'])
        else:
            cur_content = str(cur_news['topic']) + '[SEP]' + str(cur_news['title']) + '[SEP]' + str(cur_news['content'])
        token_id = tokenizer.encode(cur_content, max_length=self.max_len, truncation=True, add_special_tokens=True, padding='max_length')
        token_id = torch.tensor(token_id)
        mask_token_id = tokenizer.pad_token_id
        mask = (token_id != mask_token_id)

        return newsid, token_id, mask
    
    def __len__(self):
        return len(self.news_base)

def word2input(texts, max_len):
    token_ids = []
    for text in texts:
        token_id = tokenizer.encode(text, 
                                    max_length=max_len, 
                                    padding='max_length', 
                                    truncation=True,
                                    add_special_tokens=True)
        token_ids.append(token_id)
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)

    return token_ids, masks

