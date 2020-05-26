'''
Util functions for building dataset
'''
import os
import time
import json
import torch
import string
import random
import logging
import ijson
import argparse
import nltk
import numpy as np
from tqdm import tqdm 
from multiprocessing import Manager
from torch.utils.data import Dataset
from nltk.stem.porter import PorterStemmer
from pytorch_pretrained_bert import BertTokenizer, BertModel
from config import DATA_PATH, START_TAG, STOP_TAG, tag_to_ix, BERT_HIDDEN
import config as cfg

stemmer = PorterStemmer()
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------
# load datasets or preprocess features
# -------------------------------------------------------------------------------------------
def read_openkp_examples(args, tokenizer):
    ''' load preprocess cached_features files. '''            
    if not os.listdir(args['data_path']):
        logger.info('Error : not found %s' % args['data_path'])

    if args['run_mode'] == 'train':
        mode_dict = {'train':[], 'valid':[]}
    elif args['run_mode'] == 'generate':
        #mode_dict = {'eval_public':[], 'valid':[]}
        mode_dict = {'fashion':[]}
    else:
        raise Exception("Invalid run mode %s!" % args['run_mode'])
        
    for mode in mode_dict:
        filename = os.path.join(args['data_path'], "openkp.%s.json" % mode)
        logger.info("start loading openkp %s data ..." %mode)
        with open(filename, "r", encoding="utf-8") as f:
            mode_dict[mode] = json.load(f)
        f.close()
        logger.info("success loaded openkp %s data : %d " %(mode, len(mode_dict[mode])))
    return mode_dict

def extract_sentence(cur_item, all_items):
    res_item = {'tokens':[], 'labels':[], 'masks':[]}
    tokens = cur_item['tokens']
    labels = cur_item['labels']
    ending_pos = [-1] # positions of '.'
    for index in range(len(tokens)):
        if tokens[index] in ['.', '?', '!']:
            #print("Find endings!")
            ending_pos.append(index)
    
    for i in range(len(ending_pos)-1):
        pos1 = ending_pos[i]
        pos2 = ending_pos[i+1]
        if 'B' in labels[pos1:pos2]:
            #print("Find keyphrase")
            all_items.append({ 'tokens':tokens[pos1+1:pos2], 'labels':labels[pos1+1:pos2], 'masks':cur_item['masks'][pos1+1:pos2] })
        

# -------------------------------------------------------------------------------------------
# ijson: used to parse big json file
# -------------------------------------------------------------------------------------------
def parse_json(parser, mode, num_train=128):
    all_items = []
    if num_train > 134891:
        logger.error("Training samples overflow")
    if mode == 'train':
        num_train = cfg.num_train
    if mode == 'valid':
        num_train = cfg.num_test
    index = 0
    count = 0
    cur_tokens = []
    cur_masks = []
    cur_labels = []
    cur_item = {'tokens':[], 'labels':[], 'masks':[]}
    f_token = False
    f_mask = False
    f_label = False
    for prefix, event, value in parser:
        if index == num_train:
            break
        if count == 3:
            all_items.append(cur_item)
            # extract_sentence(cur_item, all_items)
            cur_item = {'tokens':[], 'labels':[], 'masks':[]}
            index+=1
            if index % 100 == 0:
                print("Loaded {} samples ".format(index))
            count = 0
            #print("Parsed item No.{}".format(index))
            cur_tokens = []
            cur_masks = []
            cur_labels = []
        if (prefix, event) == ('item.orig_tokens', 'start_array') or f_token:
            # read tokens
            f_token = True
            if (prefix, event) == ('item.orig_tokens', 'start_array'):
                continue
            if event == 'end_array':
                f_token = False
                count += 1
                cur_item['tokens'] = cur_tokens
            else:
                cur_tokens.append(value)
        if (prefix, event) == ('item.label', 'start_array') or f_label:
            # read labels
            f_label = True
            if (prefix, event) == ('item.label', 'start_array'):
                continue
            if event == 'end_array':
                f_label = False
                count += 1
                cur_item['labels'] = cur_labels
            else:
                cur_labels.append(value)
        
        if (prefix, event) == ('item.valid_mask', 'start_array') or f_mask:
            # read masks
            f_mask = True
            if (prefix, event) == ('item.valid_mask', 'start_array'):
                continue
            if event == 'end_array':
                f_mask = False
                count += 1
                cur_item['masks'] = cur_masks
            else:
                cur_masks.append(value)
        

    print("Completed Parsing json !")
    return all_items

def read_ijson_examples(args, tokenizer, dataset_type):
    ''' load data from big json file '''            
    if not os.listdir(args['data_path']):
        logger.info('Error : not found %s' % args['data_path'])
    
    if dataset_type == 'train':
        mode = 'train'
    elif dataset_type == 'valid':
        mode = 'valid'
    elif args['run_mode'] == 'generate':
        mode = 'eval_public'
    else:
        raise Exception("Invalid run mode %s!" % args['run_mode'])

    openkp_file = os.path.join(args['data_path'], "openkp.%s.json" % mode)
    f = open(openkp_file, 'r', encoding='utf8')
    print("Opened json file")
    parser = ijson.parse(f)
    all_items = parse_json(parser, mode)   
    print("mode {}, all_items length is {}".format(mode, len(all_items))) 
    return all_items

# -------------------------------------------------------------------------------------------
# build dataset and dataloader
# -------------------------------------------------------------------------------------------        
class build_openkp_dataset(Dataset):
    ''' build datasets for train & eval '''
    def __init__(self, args, dataset_type, tokenizer, converter, shuffle=False):
        #self.manager = Manager()
        self.dataset_type = dataset_type
        self.all_items = read_ijson_examples(args, tokenizer, dataset_type)
        #self.all_items = self.manager.list(self.all_items)
        self.tokenizer = tokenizer
        self.converter = converter
        if shuffle:
            random.seed(args['seed'])
            random.shuffle(self.all_items)
            
    def __len__(self):
        return len(self.all_items)

    def __getitem__(self, index):
        return convert_items_to_features(index, self.all_items[index], self.tokenizer, 
                                            self.converter, self.dataset_type)
'''
def convert_items_to_features(index, item, tokenizer, tag_to_ix, run_mode):
    num_tokens = len(item['tokens'])
    if num_tokens < cfg.MAX_LEN:
        padded_tokens = item['tokens'] + [cfg.PAD_TOKEN for i in range( cfg.MAX_LEN-num_tokens ) ] 
        src_tokens = [cfg.CLS_TOKEN] + padded_tokens + [cfg.SEP_TOKEN]
    else:   
        src_tokens = [cfg.CLS_TOKEN] + item['tokens'][:cfg.MAX_LEN] + [cfg.SEP_TOKEN]

    if run_mode == 'train' or run_mode == 'valid':
        if num_tokens < cfg.MAX_LEN:
            padded_tags = [tag_to_ix[t] for t in item['labels'] ] + [tag_to_ix['O'] for i in range( cfg.MAX_LEN-num_tokens ) ]
            targets = [tag_to_ix['O'] ] + padded_tags + [tag_to_ix['O'] ] 
        else:
            targets = [tag_to_ix['O'] ] + [tag_to_ix[t] for t in item['labels'] ][:cfg.MAX_LEN] + [tag_to_ix['O'] ] 
            
        assert len(src_tokens) == len(targets)
        targets = torch.tensor( targets, dtype=torch.long)
        return src_tokens, targets
    
    else:
        logger.info('not the mode : %s'% run_mode) 

def collate_wrapper(batch):
    src_tokens = [x[0] for x in batch]
    src_labels = [x[1] for x in batch]
    
    labels = torch.LongTensor(len(src_labels), cfg.MAX_LEN+2).zero_()
    for i, t in enumerate(src_labels):
        labels[i, :].copy_(t)

    return src_tokens, labels
'''

def convert_items_to_features(index, item, tokenizer, tag_to_ix, run_mode):
    if run_mode == 'train' or run_mode == 'valid':
        src_tokens = item['tokens'] 
        targets = [tag_to_ix[t] for t in item['labels'] ]
            
        assert len(src_tokens) == len(targets)
        return src_tokens, targets  
    else:
        logger.info('not the mode : %s'% run_mode) 

def collate_wrapper(batch):
    '''
        Alignment of mini-batch.
    '''
    src_tokens = [x[0] for x in batch]
    src_labels = [x[1] for x in batch]
    ret_tokens = []
    ret_labels = []
    max_len = max( [len(x) for x in src_labels]  )
    if max_len < cfg.MAX_LEN:
        for tokens in src_tokens:
            tokens = tokens + [cfg.PAD_TOKEN for i in range( max_len-len(tokens) ) ]
            tokens = [cfg.CLS_TOKEN] + tokens + [cfg.SEP_TOKEN]
            ret_tokens.append(tokens)

        for labels in src_labels:
            labels = labels + [cfg.tag_to_ix['O'] for i in range( max_len-len(labels) ) ]
            labels = [cfg.tag_to_ix['O'] ] + labels + [cfg.tag_to_ix['O'] ]
            ret_labels.append(labels)
    else:
        # PAD or trim to MAX_LEN
        for tokens, labels in zip(src_tokens, src_labels):
            seq_len = len(tokens)
            if seq_len < cfg.MAX_LEN:
                tokens = tokens + [cfg.PAD_TOKEN for i in range( cfg.MAX_LEN-seq_len ) ]
                tokens = [cfg.CLS_TOKEN] + tokens + [cfg.SEP_TOKEN]
                labels = labels + [cfg.tag_to_ix['O'] for i in range( cfg.MAX_LEN-seq_len ) ]
                labels = [cfg.tag_to_ix['O'] ] + labels + [cfg.tag_to_ix['O'] ]
                ret_tokens.append(tokens)
                ret_labels.append(labels)
            else: 
                tokens = [cfg.CLS_TOKEN] + tokens[:cfg.MAX_LEN] + [cfg.SEP_TOKEN]
                labels = [cfg.tag_to_ix['O'] ] + labels[:cfg.MAX_LEN] + [cfg.tag_to_ix['O'] ] 
                ret_tokens.append(tokens)
                ret_labels.append(labels) 
        
        
    tensor_labels = torch.LongTensor(len(ret_labels), len(ret_labels[0]) ).zero_()
    for i, t in enumerate(ret_labels):
        tensor_labels[i, :].copy_(torch.tensor(t))

    return ret_tokens, tensor_labels