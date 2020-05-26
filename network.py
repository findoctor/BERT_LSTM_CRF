import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer, BertModel
import config as cfg

torch.manual_seed(1)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_bert_batch_input(tokenizer, sentences, mode):
    # use the first sub-word to represent out-of-vocab word
    if mode == 'train' or mode == 'test':
        tokenized_text = [tokenizer.tokenize(item)[0] if len(tokenizer.tokenize(item) ) > 0 \
                            else cfg.UNK_TOKEN  for sentence in sentences for item in sentence]
                
    if mode == 'eval':
        sentence = [cfg.CLS_TOKEN] + sentence.split() + [cfg.SEP_TOKEN]
        tokenized_text = [tokenizer.tokenize(item)[0] for item in sentence]
     
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.view(cfg.batch_size, -1)
    return tokens_tensor

def prepare_bert_input(tokenizer, sentence, mode):
    # use the first sub-word to represent out-of-vocab word
    if mode == 'train' or mode == 'test':
        tokenized_text = [tokenizer.tokenize(item)[0] if len(tokenizer.tokenize(item) ) > 0 \
                            else cfg.UNK_TOKEN for item in sentence]
                
    if mode == 'eval':
        sentence = [cfg.CLS_TOKEN] + sentence.split() + [cfg.SEP_TOKEN]
        tokenized_text = [tokenizer.tokenize(item)[0] for item in sentence]
     
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    return tokens_tensor


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BERT_BiLSTM_CRF(nn.Module):

    def __init__(self, tag_to_ix, bert, lstm_hidden):
        super(BERT_BiLSTM_CRF, self).__init__()
        
        # Use pretrained BERT
        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert = bert
        self.hidden_dim = bert.config.hidden_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        
        self.lstm_hidden = lstm_hidden
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=lstm_hidden // 2,
                            num_layers=1, bidirectional=True, batch_first=False)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(lstm_hidden, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[cfg.START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[cfg.STOP_TAG]] = -10000
        
        self.hidden = self.init_hidden()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def init_hidden(self):
        return (torch.randn(2, 1, self.lstm_hidden // 2),
                torch.randn(2, 1, self.lstm_hidden // 2))
    
    def bert_enc(self, x):
        bert_emb, _ = self.bert(x)
        bert_enc = bert_emb[-1]
        # delete embedding for [CLS] [SEP]
        # bert_enc: [1, len, 768]
        bert_enc = bert_enc.narrow(1, 1, bert_enc.shape[1]-2)
        return bert_enc

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(self.device)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[cfg.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[cfg.STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        bert_embs = self.bert_enc(sentence)
        # bert_embds: [n_batch, seq_len, 768]
        enc, _ = self.lstm(bert_embs)
        lstm_feats = self.hidden2tag(enc)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(self.device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[cfg.START_TAG]], dtype=torch.long, device=self.device), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[cfg.STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(self.device)
        init_vvars[0][self.tag_to_ix[cfg.START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[cfg.STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[cfg.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        feats = feats.view(-1, self.tagset_size)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)
        
        # batch=1 mode
        lstm_feats = lstm_feats.view(-1, self.tagset_size)
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq