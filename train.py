import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer, BertModel
import config as cfg
from build_dataset import build_single_batch
import network_batch as network
from evaluate import evaluate
from tqdm import tqdm 
import os
import logging
from torch.utils.tensorboard import SummaryWriter

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

logger = logging.getLogger()
# -------------------------------------------------------------------------------------------
# load datasets 
# -------------------------------------------------------------------------------------------
train_data_loader, test_data_loader = build_single_batch()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("DEVICE TYPE {}".format(device))


bert_pretrained = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = network.BERT_BiLSTM_CRF( cfg.tag_to_ix, bert_pretrained, cfg.LSTM_HIDDEN)

# freeze bert pretrained gradient
for param in model.bert.parameters():
    param.requires_grad = False

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

eval_sentence_tokens = None
eval_sentence_targets = None

all_step = 0

for epoch in range(cfg.n_epoch): 
    epoch_loss = 0.0
    epoch_step = 0
    for step, batch in enumerate(tqdm(train_data_loader)):

        batch_sentence_tokens = batch[0]
        batch_targets = batch[1]

        model.zero_grad()
        model.train()   # !!!
        sentence_in = network.prepare_bert_batch_input(tokenizer, batch_sentence_tokens, 'train')

        assert sentence_in.shape == batch_targets.shape

        '''
        # delete Y label for [CLS] [SEP] in target
        for target in batch_targets:
            target = target.narrow(0, 1, target.shape[0]-2)
        '''
        
        # batch_targets = batch_targets.narrow(1, 1, batch_targets.shape[1]-2)
        sentence_in = sentence_in.to(device)
        batch_targets = batch_targets.to(device)
        
        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, batch_targets)
        epoch_loss += loss.item()  # Remember to add .item, or it will accumulate iin the memory
        epoch_step += 1
        all_step +=1
        loss.backward()
        optimizer.step()
        print("LOSS: {}".format(loss.item()))

        # Write Loss every 10 moni-batch and evaluate
        if all_step % 10 == 0:
            f1_score = evaluate(model, test_data_loader, tokenizer, device)
            writer.add_scalar('Loss/train', loss.item()/cfg.batch_size, all_step/10)
            writer.add_scalar('F1/test', f1_score, all_step/10)

        #print("GPU Usage: {}".format(torch.cuda.memory_summary(device=device) ) )       
	
    logger.info('Train Loss = %.6f (epoch = %d)'% (epoch_loss/epoch_step, epoch))
    saved_model_path = os.path.join(cfg.output_folder, 'epoch_{}.pth'.format(epoch+1))
    torch.save(model.state_dict(), saved_model_path)

