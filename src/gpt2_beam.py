#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import time
import math
import os, sys
import json
import itertools
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# import torch
# from torch import Tensor, device, dtype, nn
# from torch.nn import CrossEntropyLoss
# from torch.nn import functional as F
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# torch.set_printoptions(threshold=100000)
import jittor as jt
jt.flags.use_cuda = 1
from jittor.nn import CrossEntropyLoss
from jittor.dataset import DataLoader
from jittor import nn as jt_nn

import numpy as np

# from gpu import (
#     add_gpu_params,
#     parse_gpu,
#     distributed_opt,
#     distributed_gather,
#     distributed_sync,
#     cleanup
# )

from exp_utils import create_exp_dir

from data_utils import FT_Dataset
from model import GPT2Config, GPT2LMModel


parser = argparse.ArgumentParser(description='PyTorch GPT2 beam decoding')

# add_gpu_params(parser)

parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')

parser.add_argument('--batch_size', type=int, default=10,
                    help='batch size')

parser.add_argument('--seq_len', type=int, default=512,
                    help='number of tokens to predict')

parser.add_argument('--eval_len', type=int, default=256,
                    help='evaluation length')

parser.add_argument('--min_length', type=int, default=0,
                    help='minimum generation length')

parser.add_argument('--model_card', default='gpt2.sm', choices=['gpt2.sm', 'gpt2.md', 'gpt2.lg'],
                    help='model names')

parser.add_argument('--init_checkpoint', default=None, type=str, help='initial checkpoint')

parser.add_argument('--lora_dim', type=int, default=0, help='lora attn dimension')

parser.add_argument('--lora_alpha', type=int, default=128, help='lora attn alpha')

parser.add_argument('--work_dir', type=str, default=os.getenv('PT_OUTPUT_DIR', 'gpt2_model'),
                    help='working folder')

parser.add_argument('--beam', type=int, default=1, help='beam search size')

parser.add_argument('--length_penalty', type=float, default=1.0, help='length penalty')

parser.add_argument('--no_repeat_ngram_size', type=int, default=4, help='no_repeat_ngram_size')

parser.add_argument('--repetition_penalty', type=float, default=1.0, help='repetition_penalty')

parser.add_argument('--eos_token_id', action='append', type=int, default=[50256],
                    help='eos token id')

parser.add_argument('--output_file', type=str, default='beam_prediction.jsonl',
                    help='output file name')


def print_args(args):
    print('=' * 100)
    for k, v in args.__dict__.items():
        print('        - {} : {}'.format(k, v))
    print('=' * 100)


def _reorder_cache(past: Tuple, beam_idx: jt.array) -> Tuple[jt.array]:
    return tuple(jt.misc.index_select(layer_past, 1, beam_idx) for layer_past in past)

def _calc_banned_ngram_tokens(
    prev_input_ids: jt.array,
    num_hypos: int,
    no_repeat_ngram_size: int,
    cur_len: int
) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]

    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def _enforce_repetition_penalty_(
    lprobs,
    batch_size,
    num_beams,
    prev_output_tokens,
    repetition_penalty
):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """

    for i in range(batch_size * num_beams):
        print('prev_output_tokens.shape', prev_output_tokens.shape)
        print('prev_output_tokens[i].shape', prev_output_tokens[i].shape)

        for previous_token in set(prev_output_tokens[i].tolist()):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty

def _postprocess_next_token_scores(
    scores,
    history,
    cur_len,
    batch_size,
    num_beams,
    repetition_penalty=1.0,
    no_repeat_ngram_size=4,
    bad_words_ids=None,
    min_length=0,
    max_length=100,
    eos_token_id=None,
):
    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if repetition_penalty != 1.0 and history is not None:
        _enforce_repetition_penalty_(scores, batch_size, num_beams, history, repetition_penalty)

    # score: batch_size * beam, vocab
    # set eos token prob to zero if min_length is not reached
    if eos_token_id is not None and cur_len < min_length:
        for eos in eos_token_id:
            scores[:, eos] = -float("inf")

    if no_repeat_ngram_size > 0 and history is not None:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = _calc_banned_ngram_tokens(
                history, num_batch_hypotheses, no_repeat_ngram_size, cur_len
        )

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

    return scores


def _add_beam_candidate(
    best_score,
    best_sequence,
    batch_size,
    num_beams,
    beam_scores,
    history,
    eos_token_id=None
):
    last_tokens = history[:, -1]
    for _i in range(batch_size * num_beams):
        if eos_token_id is None or last_tokens[_i] in eos_token_id:
            cur_len = history.shape[-1]
            _score = beam_scores.reshape(-1)[_i] / cur_len ** args.length_penalty

            batch_id = _i // num_beams

            if not batch_id in best_score or best_score[batch_id] < _score:
                best_score[batch_id] = _score
                best_sequence[batch_id][:cur_len] = history[_i]

            beam_scores.reshape(-1)[_i] = -float("inf")


def beam(model, data_iter, args):
    import csv
    model.eval()
    total_loss = 0.
    start_time = time.time()

    all_predictions = {}
    time_records = []

    with jt.no_grad():
        for idx, data in enumerate(data_iter):
            data = {key: value for key, value in data.items()}

            _id = data['id']
            _query = data['query']
            _query_len = data['query_len']

            ## local adaptation start.

            ## local adaptation end.


            output = None
            score = None

            batch_size = _id.size(0)
            num_beams = args.beam
            length_penalty = args.length_penalty

            _batch = jt.arange(0, _id.size(0), dtype=jt.int64)

            past = None
            len_past = None

            _query = _query.repeat(1, num_beams).reshape(batch_size * num_beams, -1)
            _query_len = _query_len.unsqueeze(-1).repeat(1, num_beams).reshape(-1)

            _bbatch = _batch.unsqueeze(-1).repeat(1, num_beams).reshape(-1)

            # scores for each sentence in the beam
            beam_scores = jt.zeros(
                (batch_size, num_beams), dtype=jt.float32
            )

            best_sequence = jt.zeros(
                (batch_size, args.eval_len), dtype=jt.int64
            )
            best_score = {}

            history = None
            with jt.no_grad():
                for i in range(0, args.eval_len):
                    loop_start_time = time.time()
                    if i == 0:
                        logits, past = model(_query)
                        logits = logits[_bbatch, (_query_len-1).long(), :] # batch_size * beam, vocab
                    else:
                        #print('token_id.shape', token_id.shape, token_id)
                        #print('past.shape', past[0].shape)
                        #print('len_past.shape', len_past.shape, len_past)

                        logits, past = model(token_id, past=past, len_past=len_past)
                        logits = logits[:, -1, :]    # batch_size * beam, vocab

                    logits = _postprocess_next_token_scores(
                        logits,
                        history,
                        i,
                        batch_size,
                        num_beams,
                        repetition_penalty=args.repetition_penalty,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        min_length=args.min_length,
                        eos_token_id=args.eos_token_id,
                    )

                    softmax_probs = jt_nn.softmax(logits, dim=-1)
                    ##_prob, _w_idx = torch.topk(softmax_probs, num_beams) # batch_size, beam

                    vocab_size = softmax_probs.shape[-1]


                    _logprob = jt.log(softmax_probs) # batch_size * beam, vocab
                    if i == 0:
                        next_scores = _logprob.reshape(batch_size, num_beams, -1)[:, 0, :] # batch_size, vocab

                    else:
                        next_scores = beam_scores.unsqueeze(-1) + _logprob.reshape(batch_size, num_beams, -1)
                        next_scores = next_scores.reshape(batch_size, -1) # batch_size, beam * vocab
                    next_scores_time = time.time()

                    next_scores, next_tokens = jt.topk(
                        next_scores, num_beams, dim=1, largest=True, sorted=True
                    )     # batch_size, num_beams
                    topk_time = time.time()

                    beam_id = (next_tokens // vocab_size).reshape(-1)    # batch_size * num_beams
                    token_id = (next_tokens % vocab_size).reshape(-1).unsqueeze(-1) # batch_size, num_beams

                    beam_idx = beam_id.reshape(batch_size, num_beams) + (_batch * num_beams).unsqueeze(-1)
                    past = _reorder_cache(past, beam_idx.reshape(-1))
                    beam_scores = next_scores # batch_size, num_beams
                    len_past = (_query_len + i).long()
                    reorder_time = time.time()

                    if history is None:
                        history = token_id.detach()
                    else:
                        history = jt.cat((history[beam_idx.reshape(-1)], token_id.detach()), dim=1).detach()
                    history_time = time.time()

                    _add_beam_candidate(
                        best_score, best_sequence, batch_size, num_beams, beam_scores, history,
                        eos_token_id=args.eos_token_id
                    )
                    add_beam_time = time.time()
                    loop_end_time = time.time()
                    time_records.append({
                        'id': f'{idx}_{i}',
                        'next_scores_time': next_scores_time - loop_start_time,
                        'topk_time': topk_time - next_scores_time,
                        'reorder_time': reorder_time - topk_time,
                        'history_time': history_time - reorder_time,
                        'add_beam_time': add_beam_time - history_time,
                        'total_time': loop_end_time - loop_start_time
                    })

                _add_beam_candidate(
                    best_score, best_sequence, batch_size, num_beams, beam_scores, history
                )


            with jt.no_grad():
                _id = _id.data
                output = best_sequence.data

            _id = _id.reshape(-1)
            output = output.reshape(-1, output.shape[-1])
            #score = score.reshape(-1, score.shape[-1]).cpu()

            for _b in range(0, _id.shape[-1]):
                _i = int(_id[_b].item())
                all_predictions[_i] = {}
                all_predictions[_i]['id'] = _i
                all_predictions[_i]['predict'] = output[_b].tolist()
                #all_predictions[_i]['score'] = score[_b].tolist()

            if idx % 10 == 0:
                print('inference samples', idx)

    end_time = time.time()
    print(f'Total time taken for beam function: {end_time - start_time} seconds')

    # Write time_records to a CSV file
    with open('time_records.csv', 'w', newline='') as csvfile:
        fieldnames = ['id', 'next_scores_time', 'topk_time', 'reorder_time', 'history_time', 'add_beam_time', 'total_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for record in time_records:
            writer.writerow(record)

    pred_file = os.path.join(args.work_dir, args.output_file)
    print('saving prediction file', pred_file)
    with open(pred_file, 'w') as writer:
        for _i in all_predictions:
            writer.write(json.dumps(all_predictions[_i]) + '\n')


if __name__ == '__main__':
    args = parser.parse_args()
    print_args(args)

    args.logging = create_exp_dir(args.work_dir)

    valid_data = FT_Dataset(
        args.data, args.batch_size, args.seq_len, args.eval_len,
    )
    valid_sampler = jt.dataset.Sampler(valid_data)
    valid_loader = jt.dataset.DataLoader(
        valid_data, batch_size=args.batch_size, sampler = valid_sampler
    )

    if args.model_card == 'gpt2.sm':
        config = GPT2Config(
            n_embd=768, n_layer=12, n_head=12,
            lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha,
        )
    elif args.model_card == 'gpt2.md':
        config = GPT2Config(
            n_embd=1024, n_layer=24, n_head=16,
            lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha,
        )
    elif args.model_card == 'gpt2.lg':
        config = GPT2Config(
            n_embd=1280, n_layer=36, n_head=20,
            lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha,
        )

    lm_net = GPT2LMModel(config)
    if args.init_checkpoint is not None:
        print('loading model pretrained weight.')
        import pickle
        with open(args.init_checkpoint, 'rb') as f:
            cp = pickle.load(f)
        lm_net.load_weight(cp)

    print('model sampling ...')
    beam(lm_net, valid_loader, args)
    print('Sampling completed.')
