# Copyright (c) 2019, Facebook, Inc. and its affiliates. All Rights Reserved

from __future__ import absolute_import, division, print_function

from tqdm import tqdm
import pickle as pkl
import argparse
import collections
import json
import logging
import math
import os
import random
import time
import sys
import gzip
from torch import nn
from io import open

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.blanc import BLANC
from pytorch_pretrained_bert.tokenization import BasicTokenizer, BertTokenizer
from mrqa_official_eval import exact_match_score, f1_score, metric_max_over_ground_truths

PRED_FILE = "predictions.json"
EVAL_FILE = "eval_results.txt"
TEST_FILE = "test_results.txt"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class MRQAExample(object):
    """
    A single training/test example for the MRQA dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 sentences=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.sentences=sentences

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position


def read_mrqa_examples(input_file, is_training):
    """Read a MRQA json file into a list of MRQAExample."""
    with gzip.GzipFile(input_file, 'r') as reader:
        # skip header
        content = reader.read().decode('utf-8').strip().split('\n')[1:]
        input_data = [json.loads(line) for line in content]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    num_answers = 0
    for i, entry in enumerate(input_data):
        if i % 1000 == 0:
            logger.info("Processing %d / %d.." % (i, len(input_data)))
        paragraph_text = entry["context"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        sentence_start_end = [[char_to_word_offset[se[0]], char_to_word_offset[se[1]]] for se in entry['sentences']]

        for qa in entry["qas"]:
            qas_id = qa["qid"]
            question_text = qa["question"]
            start_position = None
            end_position = None
            orig_answer_text = None
            
            answers = qa["detected_answers"]
            # import ipdb
            # ipdb.set_trace()
            spans = sorted([span for spans in answers for span in spans['char_spans']])
            # take first span
            char_start, char_end = spans[0][0], spans[0][1]
            orig_answer_text = paragraph_text[char_start:char_end+1]
            start_position, end_position = char_to_word_offset[char_start], char_to_word_offset[char_end]
            num_answers += sum([len(spans['char_spans']) for spans in answers])

            example = MRQAExample(
                qas_id=qas_id,
                question_text=question_text,
                doc_tokens=doc_tokens,
                orig_answer_text=orig_answer_text,
                start_position=start_position,
                end_position=end_position,
                sentences=sentence_start_end)
            examples.append(example)
    logger.info('Num avg answers: {}'.format(num_answers / len(examples)))
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = -1
        tok_end_position = -1
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)
        
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (tok_start_position >= doc_start and
                    tok_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
                #continue
            else:
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
            
            if example_index < 0:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                        "answer: %s" % (answer_text))

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position))
            unique_id += 1

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def make_predictions(all_examples, all_features, all_results, n_best_size,
                     max_answer_length, do_lower_case, verbose_logging):
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result
    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(
            "NbestPrediction", ["text", "start_logit", "end_logit", "start_index", "end_index"])
        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)
                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    start_index=orig_doc_start,
                    end_index=orig_doc_end))

        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start_index=1, end_index=1))
        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry
    
        sc = best_non_null_entry.start_index
        ec = best_non_null_entry.end_index

        target_sentence = 0
        max_overlap = -1
        for lindex, line in enumerate(example.sentences):
            intersect = len(set([i for i in range(sc, ec + 1)]) & set([i for i in range(line[0], line[1] + 1)]))
            if  intersect > max_overlap:
                target_sentence = lindex
                max_overlap = intersect
        target_sentence = example.sentences[target_sentence]
        merged_context_indexes = [i for i in range(target_sentence[0], target_sentence[1] + 1)]
        orig_tokens = [example.doc_tokens[ci] for ci in merged_context_indexes]
        orig_text = " ".join(orig_tokens)
        
        target_entry = {}
        target_entry["text"] = orig_text
        target_entry["start_index"] = merged_context_indexes[0]
        target_entry["end_index"] = merged_context_indexes[-1]
        all_predictions[example.qas_id] = target_entry

    return all_predictions

def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    tok_text = " ".join(tokenizer.tokenize(orig_text))
    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def get_raw_scores(dataset, predictions, examples):
    answers = {}
    for example in dataset:
        for qa in example['qas']:
            answers[qa['qid']] = qa['answers']
    exact_scores = {}
    f1_scores = {}
    scores = {}
    precision_scores = {}
    recall_scores = {}
    for qid, ground_truths in answers.items():
        if qid not in predictions:
            #print('Missing prediction for %s' % qid)
            continue
        prediction = predictions[qid]['text']
        exact_scores[qid] = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)[0]
        scores[qid] = metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
        f1_scores[qid] = scores[qid][0]
        precision_scores[qid] = scores[qid][1]
        recall_scores[qid] = scores[qid][2]
    
    def get_precision(sp, ep, sr, er):
        p_span = set(list(range(sp, ep + 1)))
        r_span = set(list(range(sr, er + 1)))
        return 1.0 * len(p_span & r_span) / len(p_span)
    
    def get_recall(sp, ep, sr, er):
        p_span = set(list(range(sp, ep + 1)))
        r_span = set(list(range(sr, er + 1)))
        return 1.0 * len(p_span & r_span) / len(r_span)
        
    def get_f1(sp, ep, sr, er):
        p = get_precision(sp, ep, sr, er)
        r = get_recall(sp, ep, sr, er)
        if p < 1e-10 or r < 1e-10:
            return 0.0
        else:
            return 2.0 * p * r / (p + r)
    span_f1 = {}
    span_exact = {}
    span_precision = {}
    span_recall = {}
    for example in examples:
        qid = example.qas_id
        if qid not in predictions:
            continue
        sg = example.start_position
        eg = example.end_position
        
        sf = predictions[qid]["start_index"]
        ef = predictions[qid]["end_index"]
        if sf == None:
            sf = -1
        if ef == None:
            ef = -1
        span_f1[qid] = get_f1(sf, ef, sg, eg)
        if sf == sg and ef == eg:
            span_exact[qid] = 1.0
        else:
            span_exact[qid] = 0.0
        span_precision[qid] = get_precision(sf, ef, sg, eg)
        span_recall[qid] = get_recall(sf, ef, sg, eg)

    return exact_scores, \
            f1_scores, \
            precision_scores, \
            recall_scores, \
            span_exact, \
            span_f1, \
            span_precision, \
            span_recall


def make_eval_dict(exact_scores, f1_scores, precision={}, recall = {}, span_exact={}, span_f1={}, span_p={}, span_r={}, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores.values()) / total),
            ('f1', 100.0 * sum(f1_scores.values()) / total),
            ('precision', 100.0 * sum(precision.values()) / total),
            ('recall', 100.0 * sum(recall.values()) / total),
            ('span_exact', 100.0 * sum(span_exact.values()) / total),
            ('span_f1', 100.0 * sum(span_f1.values()) / total),
            ('span_precision', 100.0 * sum(span_p.values()) / total),
            ('span_recall', 100.0 * sum(span_r.values()) / total),
            ('total', total),
        ])
    else:
        total = len(qid_list)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ('precision', 100.0 * sum(precision_scores[k] for k in qid_list) / total),
            ('recall', 100.0 * sum(recall_scores[k] for k in qid_list) / total),
            ('span_exact', 100.0 * sum(span_exact.values()) / total),
            ('span_f1', 100.0 * sum(span_f1.values()) / total),
            ('span_precision', 100.0 * sum(span_p.values()) / total),
            ('span_recall', 100.0 * sum(span_r.values()) / total),
            ('total', total),
        ])


def evaluate(args, model, device, eval_dataset, eval_dataloader,
             eval_examples, eval_features, verbose=True):
    all_results = []
    model.eval()
    for input_ids, input_mask, segment_ids, example_indices in eval_dataloader:
        if len(all_results) % 1000 == 0:
            logger.info("Processing example: %d" % (len(all_results)))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits, _ = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, geometric_p=args.geometric_p, window_size=args.window_size, lmb=args.lmb)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))

    preds = \
        make_predictions(eval_examples, eval_features, all_results,
                         args.n_best_size, args.max_answer_length,
                         args.do_lower_case, args.verbose_logging)
    exact_raw, f1_raw, precision, recall, span_exact, span_f1, span_p, span_r = get_raw_scores(eval_dataset, preds, eval_examples)
    result = make_eval_dict(exact_raw, f1_raw, precision=precision, recall=recall, span_exact=span_exact, span_f1=span_f1, span_p=span_p, span_r=span_r)

    if verbose:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    return result, preds, None

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu: {}, 16-bits training: {}".format(
        device, n_gpu, args.fp16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    assert args.test_file is not None
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(args)

    tokenizer = BertTokenizer.from_pretrained(
        args.model, do_lower_case=args.do_lower_case)

    with gzip.GzipFile(args.test_file, 'r') as reader:
        content = reader.read().decode('utf-8').strip().split('\n')[1:]
        eval_dataset = [json.loads(line) for line in content]
    eval_examples = read_mrqa_examples(
        input_file=args.test_file, is_training=False)
    eval_features = convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=False)
    logger.info("***** Test *****")
    logger.info("  Num orig examples = %d", len(eval_examples))
    logger.info("  Num split examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
    model, pretrained_weights = BLANC.from_pretrained(args.output_dir)
    if args.fp16:
        model.half()
    model.to(device)
    result, preds, nbest_preds = \
        evaluate(args, model, device, eval_dataset,
                 eval_dataloader, eval_examples, eval_features)


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default=None, type=str, required=True)
        parser.add_argument("--output_dir", default=None, type=str, required=True,
                            help="The output directory where the model checkpoints and predictions will be written.")
        parser.add_argument("--test_file", default=None, type=str)
        parser.add_argument("--max_seq_length", default=384, type=int,
                            help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                                 "longer than this will be truncated, and sequences shorter than this will be padded.")
        parser.add_argument("--doc_stride", default=128, type=int,
                            help="When splitting up a long document into chunks, "
                                 "how much stride to take between chunks.")
        parser.add_argument("--max_query_length", default=64, type=int,
                            help="The maximum number of tokens for the question. Questions longer than this will "
                                 "be truncated to this length.")
        parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
        parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for predictions.")
        parser.add_argument("--eval_metric", default='f1', type=str)
        parser.add_argument("--n_best_size", default=20, type=int,
                            help="The total number of n-best predictions to generate in the nbest_predictions.json "
                                 "output file.")
        parser.add_argument("--max_answer_length", default=30, type=int,
                            help="The maximum length of an answer that can be generated. "
                                 "This is needed because the start "
                                 "and end predictions are not conditioned on one another.")
        parser.add_argument("--verbose_logging", action='store_true',
                            help="If true, all of the warnings related to data processing will be printed. "
                                 "A number of warnings are expected for a normal MRQA evaluation.")
        parser.add_argument("--no_cuda", action='store_true',
                            help="Whether not to use CUDA when available")
        parser.add_argument('--seed', type=int, default=42,
                            help="random seed for initialization")
        parser.add_argument('--fp16', action='store_true',
                            help="Whether to use 16-bit float precision instead of 32-bit")
        parser.add_argument('--geometric_p', type=float, default=0.3)
        parser.add_argument('--window_size', type=int, default=5)
        parser.add_argument('--lmb', type=float, default=0.5)
        args = parser.parse_args()

        main(args)
