import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer, BertModel
import config as cfg
from build_dataset import build_single_batch
import network_batch as network
from tqdm import tqdm 
import os
import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class eval_batch(nn.Module):
    """Base class for evaluation, provide method to calculate f1 score and accuracy
        Args:
            predicts: [batch, seq_len]
            labels: [batch, seq_len]
            X: [batch, seq_len]     -->  Original text
    """

    def __init__(self, score_type, device):
        super(eval_batch, self).__init__()
        self.score_type = score_type
        self.device = device
        
    def reset(self):
        self.correct_labels = 0
        self.total_labels = 0
        self.gold_count = 0
        self.guess_count = 0
        self.overlap_count = 0

    def extract_chunks(self, labels, tokens):
        """
        Example:
            labels:[0,0,1,2,3,0,0,4,0]
            tokens:[[cls], A, nice, looking, coat, made, in, China, [sep]]
        Return:
            [ "nice looking coat", "China" ]
        """
        chunks = []
        pos = 0
        seq_len = len(labels)
        while pos < seq_len:
            if labels[pos]==cfg.tag_to_ix['O']:
                pos+=1
                continue
            phrase_tokens = []
            while pos < seq_len and labels[pos]!=cfg.tag_to_ix['O']:
                phrase_tokens.append(tokens[pos])
                pos+=1
            phrase = " ".join(item for item in phrase_tokens)
            chunks.append(phrase)
        return set(chunks)
    
    def get_valid_length(self, tokens):
        # So as to delete PAD
        pad_index = -1
        for index in range(len(tokens)):
            if tokens[index] == cfg.PAD_TOKEN:
                pad_index = index 
                break 
        return pad_index
            
    
    def eval_instance(self, best_path, gold, tokens):
        """
        update statics for one instance
        args:
            best_path (seq_len): predicted  List
            gold (seq_len): ground-truth    tensor
        """
        total_labels = len(best_path)
        assert total_labels > 1
        best_path = torch.LongTensor(best_path).to(self.device)
        correct_labels = (best_path==gold).sum().to(self.device)
        gold_chunks = self.extract_chunks(gold, tokens)
        gold_count = len(gold_chunks)
        guess_chunks = self.extract_chunks(best_path, tokens)
        guess_count = len(guess_chunks)
        overlap_count = len(gold_chunks & guess_chunks)

        return total_labels, correct_labels, gold_count, guess_count, overlap_count

    
    def batch_f1_score(self, batch_tokens, batch_predicts, batch_labels):
        for best_path, gold, tokens in zip(batch_predicts, batch_labels, batch_tokens):
            #gold = gold[1:-1]
            #tokens = tokens[1:-1]
            valid_len = self.get_valid_length(tokens)
            if valid_len != -1:
                tokens = tokens[:valid_len]
                gold = gold[:valid_len]
                best_path = best_path[:valid_len]
            # gold = gold.numpy()
            # best_path = np.array(best_path)

            total_labels_i, correct_labels_i, gold_count_i, guess_count_i, overlap_count_i = \
                self.eval_instance(best_path, gold, tokens)
            self.total_labels += total_labels_i
            self.correct_labels += correct_labels_i
            self.gold_count += gold_count_i
            self.guess_count += guess_count_i
            self.overlap_count += overlap_count_i
    
    def get_f1_score(self, batch_tokens, batch_predicts, batch_labels):
        self.reset()
        self.batch_f1_score(batch_tokens, batch_predicts, batch_labels)
        if self.guess_count == 0:
            return 0.0
        precision = self.overlap_count / float(self.guess_count)
        recall = self.overlap_count / float(self.gold_count)
        if precision == 0.0 or recall == 0.0:
            return 0.0
        return  2 * (precision * recall) / (precision + recall)


        


def evaluate(model, test_data_loader, tokenizer, device):
    eval_class = eval_batch('f1', device).to(device)
    model.eval()
    avg_f1 = 0.0
    count_batch = 0
    for step, batch in enumerate(test_data_loader):
        count_batch += 1
        X = batch[0]
        Y = batch[1]
        
        X_input = network.prepare_bert_batch_input(tokenizer, X, 'valid').to(device)
        Y = Y.to(device)
        _, best_seqs = model(X_input)
        
        f1_score = eval_class.get_f1_score(X, best_seqs, Y)
        avg_f1 += f1_score
    avg_f1 /= count_batch
    print("*************** F1 score is {} *************".format( avg_f1 ))
    return avg_f1
        
        

