import os
import json
import torch
import logging
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer, BertModel
from utils import build_openkp_dataset, collate_wrapper
from config import args, tag_to_ix, batch_size

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
logger = logging.getLogger(__name__)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def build_train_dataLoader(split, last_iters):
    print(" ==================== start loading datasets for training ====================")
    train_dataset = build_openkp_dataset(args, 'train', tokenizer, tag_to_ix, split, last_iters)
    print("********************** Train set Len *******************{}".format(len(train_dataset)))
    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=collate_wrapper,
        pin_memory=True
    )
    return train_dataset.last_iters, train_data_loader

def build_test_dataLoader(last_iters):
    print("==================== start loading datasets for testing ====================")
    test_dataset = build_openkp_dataset(args, 'valid', tokenizer, tag_to_ix, 0, last_iters)    
    print("********************** Test set Len *******************{}".format(len(test_dataset)))
    
    test_sampler = torch.utils.data.sampler.RandomSampler(test_dataset) 

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=0,
        collate_fn=collate_wrapper,
        pin_memory=True
    )

    return test_data_loader
