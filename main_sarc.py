import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

import random
import pdb
import wandb

import data.data_sarc as data
import model

from utils import batchify, batchify_sarc, get_sarc_batch, repackage_hidden

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='dataset/SARC/2.0/main',
                    help='location of the data corpus')
parser.add_argument('--ftrain', type=str, default="train-balanced.csv")
parser.add_argument('--ftest', type=str, default="test-balanced.csv")
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--chunk_size', type=int, default=10,
                    help='number of units per chunk')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--sarc-loss', type=float, default=1,
                    help="Weight for sarc cls loss. i.e. loss = raw_loss + args.sarc_loss * sarc_bce_loss")
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=40, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.5,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.4,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str, default=randomhash + '.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str, default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--finetuning', type=int, default=500,
                    help='When (which epochs) to switch to finetuning')
parser.add_argument('--philly', action='store_true',
                    help='Use philly cluster')
args = parser.parse_args()
args.tied = True

wandb.init(
    project="cs224u",
    name=args.save
)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)


###############################################################################
# Load data
###############################################################################

def model_save(fn):
    if args.philly:
        fn = os.path.join(os.environ['PT_OUTPUT_DIR'], fn)
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)


def model_load(fn):
    global model, criterion, optimizer
    if args.philly:
        fn = os.path.join(os.environ['PT_OUTPUT_DIR'], fn)
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)


import os
import hashlib

fn = 'dataset/corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if args.philly:
    fn = os.path.join(os.environ['PT_OUTPUT_DIR'], fn)
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.SARCCorpus(args.data, args.ftrain, args.ftest)
    torch.save(corpus, fn)

# eval_batch_size = 10
test_batch_size = 1
train_data = batchify_sarc(corpus.train, args.batch_size, args)
test_data = batchify_sarc(corpus.test, test_batch_size, args)
print("# train_data:", len(train_data))
print("# test_data:", len(test_data))


###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss

criterion = None
if args.sarc_loss:
    sarc_criterion = nn.BCEWithLogitsLoss().cuda()

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.chunk_size, args.nlayers,
                       args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
wandb.watch(model)

###
if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        for rnn in model.rnn.cells:
            rnn.hh.dropout = args.wdrop
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
###

if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)


###############################################################################
# Training code
###############################################################################

# def evaluate(data_source, batch_size):
def evaluate(batch_size=1):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    data_source = test_data
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)

    n_correct = 0
    indices = list(range(len(data_source)))
    for i in indices:
        if i%50 == 0:
          print(i)
        data, targets = get_sarc_batch(data_source, i, args, evaluation=True)

        output, hidden = model(data, hidden)
        # curr_loss = len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        # total_loss += curr_loss
        
        # sarcasm classification loss
        if args.sarc_loss > 0:
            logits = model.sarc_classifier(output[-1:])
            # curr_loss = args.sarc_loss * sarc_criterion(logits.view(-1), targets[-1:].type(torch.cuda.FloatTensor))
            # total_loss += curr_loss
            curr_correct = (torch.round(logits.view(-1)) == targets[-1:].type(torch.cuda.FloatTensor)).sum()
            n_correct += curr_correct.item()

        hidden = repackage_hidden(hidden)
        torch.cuda.empty_cache()

    acc = n_correct / len(data_source)
    avg_loss = total_loss.item() / len(data_source)
    print('Eval accuracy:', acc)
    wandb.log({'AccEval':acc, 'LossEval': avg_loss})
    return avg_loss


def train():
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch = 0
    indices = list(range(len(train_data)))
    random.shuffle(indices)

    cnt = 0
    accum_loss = 0
    for i in indices:
        cnt += 1
        seq_len = 1

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_sarc_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        # output, hidden = model(data, hidden, return_h=False)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        loss = raw_loss
        
        if args.sarc_loss > 0:
            logits = model.sarc_classifier(output[-1:])
            loss += args.sarc_loss * sarc_criterion(logits.view(-1), targets[-1:].type(torch.cuda.FloatTensor))


        # Activiation Regularization
        if args.alpha:
            loss = loss + sum(
                args.alpha * dropped_rnn_h.pow(2).mean()
                for dropped_rnn_h in dropped_rnn_hs[-1:]
            )
        # Temporal Activation Regularization (slowness)
        if args.beta:
            loss = loss + sum(
                args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean()
                for rnn_h in rnn_hs[-1:]
            )
        loss.backward()
        accum_loss += loss.item()

        if cnt % 50 == 0:
            wandb.log({'LossTrain': accum_loss / 50})
            cnt = 0
            accum_loss = 0

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1


# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0, 0.999), eps=1e-9, weight_decay=args.wdecay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, patience=2, threshold=0)
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            # val_loss2 = evaluate(val_data, eval_batch_size)
            # val_loss2 = evaluate(test_data, test_batch_size)
            val_loss2 = evaluate()
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
            print('-' * 89)

            if val_loss2 < stored_loss:
                model_save(os.path.join('ckpt', args.save))
                print('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

            if epoch == args.finetuning:
                print('Switching to finetuning')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
                best_val_loss = []

            if epoch > args.finetuning and len(best_val_loss) > args.nonmono and val_loss2 > min(
                    best_val_loss[:-args.nonmono]):
                print('Done!')
                import sys

                sys.exit(1)

        else:
            # val_loss = evaluate(val_data, eval_batch_size)
            # val_loss = evaluate(test_data, test_batch_size)
            val_loss = evaluate()
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 89)

            if val_loss < stored_loss:
                model_save(os.path.join('ckpt', args.save))
                print('Saving model (new best validation)')
                stored_loss = val_loss

            if args.optimizer == 'adam':
                scheduler.step(val_loss)

            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (
                    len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                print('Switching to ASGD')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if epoch in args.when:
                print('Saving model before learning rate decreased')
                model_save(os.path.join('ckpt', '{}.e{}'.format(args.save, epoch)))
                print('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)

        print("PROGRESS: {}%".format((epoch / args.epochs) * 100))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(os.path.join('ckpt', args.save))

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)
