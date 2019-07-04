
import json_lines
import logging
from my_utils import pad_sequence

class Document:
    def __init__(self):
        self.context = ''
        self.context_token = [] # list[list], [[token1, start_offset],[token2, start_offset]]
        self.qas = []

class QA:

    def __init__(self):
        self.answers = [] # list, [str1, str2]
        self.question = ''
        self.id = ''
        self.qid = ''
        self.question_tokens = [] # list[list], [[token1, start_offset], [token2, start_offset]]
        self.detected_answers = [] # list[list], [{text, char_spans, token_spans}]
        # char_spans [[start, end]], token_spans [[start, end]]

def load_data(train_jsonl_file, data_type): # data_type=[train, dev, test]
    doc_num = 0
    qa_num = 0

    documents = []
    question_answer = {}
    dataset = {"documents": documents, 'question_answer':question_answer}
    with json_lines.open(train_jsonl_file) as f:
        for item in f:

            if 'header' in item:
                dataset['name'] = item['header']['dataset']

            if 'qas' in item:

                document = Document()
                document.context = item['context']
                document.context_token = item['context_tokens']
                qas = item['qas']

                for qa in qas:
                    qa_ = QA()

                    qa_.question = qa['question']
                    # qa_.id = qa['id'] some datasets don't have id
                    qa_.qid = qa['qid']
                    qa_.question_tokens = qa['question_tokens']
                    if data_type != 'test':
                        qa_.answers = qa['answers']
                        qa_.detected_answers = qa['detected_answers']
                        question_answer[qa['qid']] = qa['answers']
                    document.qas.append(qa_)
                    qa_num += 1

                documents.append(document)
                doc_num += 1

    logging.info("{}: {} documents, {} questions".format(train_jsonl_file, doc_num, qa_num))
    return dataset


def prepare_instance(dataset, opt, tokenizer, data_type):
    dataset_instances = []

    stop = False

    documents = dataset['documents']
    instance_to_document_index = []
    for document_idx, document in enumerate(documents):

        if stop:
            break

        for qa in document.qas:

            question_word_pieces = []
            for token in qa.question_tokens:
                word_pieces = tokenizer.tokenize(token[0])
                for word_piece in word_pieces:
                    question_word_pieces.append(word_piece)

            if len(question_word_pieces) > opt.max_query_length:
                question_word_pieces = question_word_pieces[0:opt.max_query_length]

            wp_to_orig_index = []
            orig_to_wp_index = []
            context_word_pieces = []
            for token_idx, token in enumerate(document.context_token):
                orig_to_wp_index.append(len(context_word_pieces))
                word_pieces = tokenizer.tokenize(token[0])
                for word_piece in word_pieces:
                    wp_to_orig_index.append(token_idx)
                    context_word_pieces.append(word_piece)

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_context = opt.max_seq_length - len(question_word_pieces) - 3
            if len(context_word_pieces) > max_tokens_for_context:
                context_word_pieces = context_word_pieces[0:max_tokens_for_context]

            tokens = []
            tokens.append('[CLS]')
            tokens.extend(question_word_pieces)
            tokens.append('[SEP]')
            segments = [0] * len(tokens)
            tokens.extend(context_word_pieces)
            tokens.append('[SEP]')
            segments.extend([1] * (len(context_word_pieces)+1))

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            mask = [1] * len(tokens)
            offset = len(question_word_pieces) + 2

            if data_type != 'test':
                # we use the first answer to train the model
                # detected_answer = qa.detected_answers[0]
                # token_span = detected_answer['token_spans'][0] # why is token_spans a list? we use the first span.
                # start_position, end_position = token_span[0], token_span[1]

                # we use the max-span answer to train the model
                max_span = -1
                max_span_idx = -1
                for detected_answer_idx, detected_answer in enumerate(qa.detected_answers):
                    token_span = detected_answer['token_spans'][0]
                    if token_span[1]-token_span[0] > max_span:
                        max_span = token_span[1]-token_span[0]
                        max_span_idx = detected_answer_idx
                detected_answer = qa.detected_answers[max_span_idx]
                token_span = detected_answer['token_spans'][0]
                start_position, end_position = token_span[0], token_span[1]

                wp_start_position = orig_to_wp_index[start_position]
                if wp_start_position >= len(context_word_pieces): # sequence may be truncated, so we need to check this.
                    continue # if start_position is out of boundary, we ignore this instance

                if end_position < len(orig_to_wp_index)-1:
                    wp_end_position = orig_to_wp_index[end_position + 1] - 1 # a word become several word pieces
                elif end_position == len(orig_to_wp_index)-1:
                    wp_end_position = len(context_word_pieces)-1

                if wp_end_position >= len(context_word_pieces):
                    wp_end_position = len(context_word_pieces) - 1

                real_start_position = wp_start_position + offset
                real_end_position = wp_end_position + offset
            else:
                real_start_position = -1
                real_end_position = -1



            instance = {'tokens': tokens, 'input_ids':input_ids, 'mask':mask, 'segments':segments,
                        'start_position':real_start_position, 'end_position':real_end_position,
                        'wp_to_orig_index':wp_to_orig_index, 'offset':offset, 'qid':qa.qid}

            if data_type=='train' and len(dataset_instances) >= opt.train_sample_size:
                stop = True
                break
            elif data_type=='dev' and len(dataset_instances) >= opt.dev_sample_size:
                stop = True
                break

            dataset_instances.append(instance)
            instance_to_document_index.append(document_idx)

    logging.info("{} instances in dataset {}".format(len(dataset_instances), dataset['name']))
    dataset_instances_dict = {'dataset_instances':dataset_instances, 'instance_to_document_index':instance_to_document_index}
    return dataset_instances_dict


import torch
from options import opt
def my_collate(x):
    input_ids = [x_['input_ids'] for x_ in x]
    mask = [x_['mask'] for x_ in x]
    segments = [x_['segments'] for x_ in x]
    start_position = [x_['start_position'] for x_ in x]
    end_position = [x_['end_position'] for x_ in x]

    lengths = [len(x_['tokens']) for x_ in x]
    max_len = max(lengths)

    input_ids = pad_sequence(input_ids, max_len)
    mask = pad_sequence(mask, max_len)
    segments = pad_sequence(segments, max_len)
    start_position = torch.LongTensor(start_position)
    end_position = torch.LongTensor(end_position)

    if opt.gpu >= 0 and torch.cuda.is_available():
        input_ids = input_ids.cuda(opt.gpu)
        mask = mask.cuda(opt.gpu)
        segments = segments.cuda(opt.gpu)
        start_position = start_position.cuda(opt.gpu)
        end_position = end_position.cuda(opt.gpu)


    return input_ids, mask, segments, start_position, end_position


def evaluate(datasets, datasets_instances, data_loaders, model, data_type):

    all_metrics = []
    all_pred_answers = []

    for idx, dataset in enumerate(datasets):

        documents = datasets[idx]['documents']
        if data_type != 'test':
            gold_answers = datasets[idx]['question_answer']
        dataset_instances = datasets_instances[idx]['dataset_instances']
        instance_to_document_index = datasets_instances[idx]['instance_to_document_index']
        data_loader = data_loaders[idx]
        data_iter = iter(data_loader)
        num_iter = len(data_loader)
        pred_answers = {}

        with torch.no_grad():
            model.eval()

            instance_start = 0
            for i in range(num_iter):
                input_ids, mask, segments, start_position, end_position = next(data_iter)
                start_logits, end_logits = model.forward(input_ids, segments, mask)

                actual_batch_size = input_ids.size(0)
                for batch_idx in range(actual_batch_size):

                    instance_start_logits = start_logits[batch_idx]
                    instance_end_logits = end_logits[batch_idx]

                    _, start_position = torch.max(instance_start_logits, 0)
                    start_position = start_position.item()
                    _, end_position = torch.max(instance_end_logits, 0)
                    end_position = end_position.item()

                    instance = dataset_instances[instance_start+batch_idx]
                    wp_to_orig_index = instance['wp_to_orig_index']
                    offset = instance['offset']

                    context_start_position = start_position - offset
                    context_end_position = end_position - offset

                    # context_start_position and context_end_position must be within the context
                    if context_start_position < 0:
                        context_start_position = 0
                    elif context_start_position >= len(wp_to_orig_index):
                        context_start_position = len(wp_to_orig_index)-1

                    if context_end_position < 0:
                        context_end_position = 0
                    elif context_end_position >= len(wp_to_orig_index):
                        context_end_position = len(wp_to_orig_index)-1

                    token_start_position = wp_to_orig_index[context_start_position]
                    token_end_position = wp_to_orig_index[context_end_position]

                    document = documents[instance_to_document_index[instance_start+batch_idx]]

                    if token_start_position <= token_end_position:
                        if token_start_position + opt.max_answer_length < token_end_position:
                            token_end_position = token_start_position + opt.max_answer_length

                        token_span = document.context_token[token_start_position:token_end_position+1]
                    else: # token_start_position > token_end_position
                        if token_end_position + opt.max_answer_length < token_start_position:
                            token_start_position = token_end_position + opt.max_answer_length
                        token_span = document.context_token[token_end_position:token_start_position+1]

                    pred_answer = ''
                    for token in token_span:
                        pred_answer += token[0]+" "
                    pred_answers[instance['qid']] = pred_answer.strip()

                instance_start += actual_batch_size

        all_pred_answers.append(pred_answers)

        if data_type != 'test':
            import mrqa_official_eval
            metrics = mrqa_official_eval.evaluate(gold_answers, pred_answers, opt.skip_no_answer)
            logging.info("evaluate on %s: exact_match %.4f, f1 %.4f" % (dataset['name'], metrics['exact_match'], metrics['f1']))
            all_metrics.append(metrics)


    if data_type != 'test':
        macro_exact_match = 0
        macro_f1 = 0
        for metrics in all_metrics:
            macro_exact_match += metrics['exact_match']
            macro_f1 += metrics['f1']
        macro_exact_match = macro_exact_match/len(all_metrics)
        macro_f1 = macro_f1/len(all_metrics)

    else:
        macro_exact_match = 0
        macro_f1 = 0


    return macro_exact_match, macro_f1, all_pred_answers

import json
import codecs
import os
def dump_results(datasets, all_pred_answers, dump_dir):

    for idx, dataset in enumerate(datasets):

        pred_labels = all_pred_answers[idx]
        with codecs.open(os.path.join(dump_dir, dataset['name']+".json"), 'w', 'UTF-8') as fp:
            fp.write(json.dumps(pred_labels, indent=4))



