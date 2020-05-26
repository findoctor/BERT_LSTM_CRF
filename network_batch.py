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
    tokenized_text = []
    if mode == 'train' or mode == 'valid':
        tokenized_text = [tokenizer.tokenize(item)[0] if len(tokenizer.tokenize(item) ) > 0 \
                            else cfg.UNK_TOKEN  for sentence in sentences for item in sentence]
                
    elif mode == 'eval':
        sentence = [cfg.CLS_TOKEN] + sentence.split() + [cfg.SEP_TOKEN]
        tokenized_text = [tokenizer.tokenize(item)[0] for item in sentence]
    else:
        raise Exception("Invalid run mode")
     
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

    def __init__(self, tag_to_ix, bert, lstm_hidden, batch_size = cfg.batch_size):
        super(BERT_BiLSTM_CRF, self).__init__()

        self.batch_size = batch_size
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
        # bert_enc: [batch_size, len, 768]
        # bert_enc = bert_enc.narrow(1, 1, bert_enc.shape[1]-2)
        return bert_enc
    '''
    def _forward_alg(self, feats):
        batch_size, seq_len, target_size = feats.shape
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((batch_size, self.tagset_size), -10000.).to(self.device)
        # START_TAG has all of the score.
        init_alphas[:, self.tag_to_ix[cfg.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
        # forward algo: compute alpha_t iteratively
        for i in range(seq_len):
            feat = feats[:,i,:].view(batch_size, target_size)
            forward_var = torch.mm(forward_var.view(-1,target_size), self.transitions) + feat  # [B,target_size]
            forward_var = torch.log(forward_var)  # in log space 
        forward_var = forward_var + self.transitions[self.tag_to_ix[cfg.STOP_TAG]]
        alpha = forward_var.sum(1)  # [B,1]
        return alpha
    '''
    def _forward_alg(self, emissions):
        """Compute the partition function in log-space using the forward-algorithm.
        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
        Returns:
            torch.Tensor: the partition scores for each batch.
                Shape of (batch_size,)
        """
        batch_size, seq_length, nb_labels = emissions.shape

        # in the first iteration, BOS will have all the scores
        alphas = self.transitions[:, self.tag_to_ix[cfg.START_TAG]].unsqueeze(0) + emissions[:, 0]

        for i in range(1, seq_length):
            alpha_t = []

            for tag in range(nb_labels):

                # get the emission for the current tag
                e_scores = emissions[:, i, tag]

                # broadcast emission to all labels
                # since it will be the same for all previous tags
                # (bs, nb_labels)
                e_scores = e_scores.unsqueeze(1)

                # transitions from something to our tag
                t_scores = self.transitions[tag, :]

                # broadcast the transition scores to all batches
                # (bs, nb_labels)
                t_scores = t_scores.unsqueeze(0)

                # combine current scores with previous alphas
                # since alphas are in log space (see logsumexp below),
                # we add them instead of multiplying
                scores = e_scores + t_scores + alphas

                # add the new alphas for the current tag
                alpha_t.append(torch.logsumexp(scores, dim=1))

            # create a torch matrix from alpha_t
            # (bs, nb_labels)
            new_alphas = torch.stack(alpha_t).t()
            alphas = new_alphas

        # add the scores for the final transition
        last_transition = self.transitions[self.tag_to_ix[cfg.STOP_TAG], :]
        end_scores = alphas + last_transition.unsqueeze(0)

        # return a *log* of sums of exps
        return torch.logsumexp(end_scores, dim=1)
        


    def _get_lstm_features(self, sentence):
        bert_embs = self.bert_enc(sentence)
        # bert_embds: [n_batch, seq_len, 768]
        enc, _ = self.lstm(bert_embs)
        lstm_feats = self.hidden2tag(enc)
        return lstm_feats

    def _compute_scores(self, emissions, tags, mask = None):
        """Compute the scores for a given batch of emissions with their tags.

        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            tags (Torch.LongTensor): (batch_size, seq_len)

        Returns:
            torch.Tensor: Scores for each batch.
                Shape of (batch_size,)
        """
        batch_size, seq_length = tags.shape
        scores = torch.zeros(batch_size).to(self.device)
        if mask is None:
            mask = torch.ones(tags.shape).to(self.device)

        # save first and last tags to be used later
        first_tags = tags[:, 0]
        last_valid_idx = mask.int().sum(1) - 1
        last_tags = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze().to(self.device)

        t_scores = self.transitions[first_tags, cfg.tag_to_ix[cfg.START_TAG]]
        e_scores = emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze().to(self.device)

        # the scores for a word is just the sum of both scores
        scores += e_scores + t_scores

        # now lets do this for each remaining word
        for i in range(1, seq_length):
            
            previous_tags = tags[:, i - 1]
            current_tags = tags[:, i]

            # calculate emission and transition scores as we did before
            e_scores = emissions[:, i].gather(1, current_tags.unsqueeze(1)).squeeze()
            t_scores = self.transitions[current_tags, previous_tags]

            scores += e_scores + t_scores

        # add the transition from the end tag to the EOS tag for each batch
        scores += self.transitions[cfg.tag_to_ix[cfg.STOP_TAG], last_tags]

        return scores

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        # feats: [batch_size, seq_len, 7]
        # tags:  [batch_size, seq_len]
        batch_size = feats.shape[0]
        seq_len = feats.shape[1]
        target_size = feats.shape[2]
        score = torch.zeros(batch_size).to(self.device)
        tags = torch.cat( (torch.tensor([self.tag_to_ix[cfg.START_TAG]], dtype=torch.long).repeat(batch_size,1).to(self.device), tags), dim=1 ).to(self.device)
        
        for j in range(batch_size):
            sentence_feat = feats[j].view(seq_len, target_size)
            tag = torch.squeeze(tags[j])
            for i in range(seq_len):
                feat = torch.squeeze(sentence_feat[i])
                #print("CHECK1 {}".format(tag.shape))
                score[j] = score[j] + self.transitions[tag[i+1], tag[i]] + feat[tag[i+1]]
        for i in range(batch_size):
            score[i] = score[i] + self.transitions[self.tag_to_ix[cfg.STOP_TAG], tags[i,-1]]
        return score
    
    def decode(self, emissions, mask=None):
        """Find the most probable sequence of labels given the emissions using
        the Viterbi algorithm.
        Args:
            emissions: (batch_size, seq_len, nb_labels)
            mask: (batch_size, seq_len) 
        Returns:
            torch.Tensor: the viterbi score for the for each batch.
                Shape of (batch_size,)
            list of lists: the best viterbi sequence of labels for each batch.
        """
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float)

        scores, sequences = self._viterbi_decode(emissions, mask)
        return scores, sequences

    def _viterbi_decode(self, emissions, mask):
        """Compute the viterbi algorithm to find the most probable sequence of labels
        given a sequence of emissions.
        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            mask (Torch.FloatTensor): (batch_size, seq_len)
        Returns:
            torch.Tensor: the viterbi score for the for each batch.
                Shape of (batch_size,)
            list of lists of ints: the best viterbi sequence of labels for each batch
        """
        batch_size, seq_length, nb_labels = emissions.shape

        # in the first iteration, BOS will have all the scores and then, the max
        alphas = self.transitions[:,self.tag_to_ix[cfg.START_TAG]].unsqueeze(0) + emissions[:, 0]
        alphas = alphas.to(self.device)
        backpointers = []

        for i in range(1, seq_length):
            alpha_t = []
            backpointers_t = []

            for tag in range(nb_labels):
                # get the emission for the current tag and broadcast to all labels
                e_scores = emissions[:, i, tag]
                e_scores = e_scores.unsqueeze(1)
                # transitions from something to our tag and broadcast to all batches
                t_scores = self.transitions[:, tag]
                t_scores = t_scores.unsqueeze(0)

                # combine current scores with previous alphas
                scores = e_scores + t_scores + alphas

                # so far is exactly like the forward algorithm,
                # but now, instead of calculating the logsumexp,
                # we will find the highest score and the tag associated with it
                max_score, max_score_tag = torch.max(scores, dim=-1)

                # add the max score for the current tag
                alpha_t.append(max_score)

                # add the max_score_tag for our list of backpointers
                backpointers_t.append(max_score_tag)

            # create a torch matrix from alpha_t
            # (bs, nb_labels)
            new_alphas = torch.stack(alpha_t).t().to(self.device)

            # set alphas if the mask is valid, otherwise keep the current values
            is_valid = mask[:, i].unsqueeze(-1).to(self.device)
            alphas = is_valid * new_alphas + (1 - is_valid) * alphas

            # append the new backpointers
            backpointers.append(backpointers_t)

        # add the scores for the final transition
        last_transition = self.transitions[self.tag_to_ix[cfg.STOP_TAG], :]
        end_scores = alphas + last_transition.unsqueeze(0)

        # get the final most probable score and the final most probable tag
        max_final_scores, max_final_tags = torch.max(end_scores, dim=1)

        # find the best sequence of labels for each sample in the batch
        best_sequences = []
        emission_lengths = mask.int().sum(dim=1)
        for i in range(batch_size):

            # recover the original sentence length for the i-th sample in the batch
            sample_length = emission_lengths[i].item()

            # recover the max tag for the last timestep
            sample_final_tag = max_final_tags[i].item()

            # limit the backpointers until the last but one
            # since the last corresponds to the sample_final_tag
            sample_backpointers = backpointers[: sample_length - 1]

            # follow the backpointers to build the sequence of labels
            sample_path = self._find_best_path(i, sample_final_tag, sample_backpointers)

            # add this path to the list of best sequences
            best_sequences.append(sample_path)
            
        return max_final_scores, best_sequences

    def _find_best_path(self, sample_id, best_tag, backpointers):
        """Auxiliary function to find the best path sequence for a specific sample.
            Args:
                sample_id (int): sample index in the range [0, batch_size)
                best_tag (int): tag which maximizes the final score
                backpointers (list of lists of tensors): list of pointers with
                shape (seq_len_i-1, nb_labels, batch_size) where seq_len_i
                represents the length of the ith sample in the batch
            Returns:
                list of ints: a list of tag indexes representing the bast path
        """

        # add the final best_tag to our best path
        best_path = [best_tag]

        # traverse the backpointers in backwards
        for backpointers_t in reversed(backpointers):

            # recover the best_tag at this timestep
            best_tag = backpointers_t[best_tag][sample_id].item()

            # append to the beginning of the list so we don't need to reverse it later
            best_path.insert(0, best_tag)

        return best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        
        forward_score = self._forward_alg(feats)
        # gold_score = self._score_sentence(feats, tags)
        gold_score = self._compute_scores(feats, tags)
        
        #return torch.sum(forward_score - gold_score) / self.batch_size
        return torch.sum(forward_score - gold_score)

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self.decode(lstm_feats)
        return score, tag_seq