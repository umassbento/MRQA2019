
import logging
import random
import numpy as np
import torch
from options import opt
from my_utils import makedir_and_clear, MyDataset
from preprocess import load_data, prepare_instance, my_collate, evaluate, dump_results
from pytorch_pretrained_bert import BertTokenizer, BertAdam
from models import BertForQuestionAnswering
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os

if __name__ == "__main__":

    logger = logging.getLogger()
    if opt.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logging.info(opt)

    if opt.random_seed != 0:
        random.seed(opt.random_seed)
        np.random.seed(opt.random_seed)
        torch.manual_seed(opt.random_seed)
        torch.cuda.manual_seed_all(opt.random_seed)

    if opt.whattodo == 'train':
        makedir_and_clear(opt.save)

        logging.info('loading training data...')
        train_datasets = []
        for train_jsonl_file in opt.train:
            train_dataset = load_data(train_jsonl_file, 'train')
            train_datasets.append(train_dataset)

        logging.info('loading dev indomain data...')
        dev_indomain_datasets = []
        for dev_jsonl_file in opt.dev_indomain:
            dev_dataset = load_data(dev_jsonl_file, 'dev')
            dev_indomain_datasets.append(dev_dataset)

        logging.info('loading dev outdomain data...')
        dev_outdomain_datasets = []
        for dev_jsonl_file in opt.dev_outdomain:
            dev_dataset = load_data(dev_jsonl_file, 'dev')
            dev_outdomain_datasets.append(dev_dataset)

        wp_tokenizer = BertTokenizer.from_pretrained(opt.bert_dir, do_lower_case=opt.do_lower_case)

        train_datasets_instances = []
        for train_dataset in train_datasets:
            train_dataset_instances = prepare_instance(train_dataset, opt, wp_tokenizer, 'train')
            train_datasets_instances.append(train_dataset_instances)

        dev_indomain_datasets_instances = []
        for dev_dataset in dev_indomain_datasets:
            dev_dataset_instances = prepare_instance(dev_dataset, opt, wp_tokenizer, 'dev')
            dev_indomain_datasets_instances.append(dev_dataset_instances)

        dev_outdomain_datasets_instances = []
        for dev_dataset in dev_outdomain_datasets:
            dev_dataset_instances = prepare_instance(dev_dataset, opt, wp_tokenizer, 'dev')
            dev_outdomain_datasets_instances.append(dev_dataset_instances)


        if opt.gpu >= 0 and torch.cuda.is_available():
            device = torch.device("cuda", opt.gpu)
        else:
            device = torch.device("cpu")
        logging.info("use device {}".format(device))

        model = BertForQuestionAnswering.from_pretrained(opt.bert_dir)
        model.to(device)

        if opt.optim == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
        elif opt.optim == "bert_adam":
            param_optimizer = list(model.named_parameters())
            param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            num_train_optimization_steps = int(
                len(train_dataset_instances['dataset_instances']) / opt.batch_size / opt.gradient_accumulation_steps) * opt.iter
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=opt.lr,
                                 warmup=opt.warmup_proportion,
                                 t_total=num_train_optimization_steps)
        else:
            raise RuntimeError("unsupported optimizer {}".format(opt.optim))

        train_data_loaders = []
        for train_dataset_instances in train_datasets_instances:
            train_loader = DataLoader(MyDataset(train_dataset_instances['dataset_instances']), opt.batch_size, shuffle=True, collate_fn=my_collate)
            train_data_loaders.append(train_loader)

        dev_indomain_data_loaders = []
        for dev_dataset_instances in dev_indomain_datasets_instances:
            dev_loader = DataLoader(MyDataset(dev_dataset_instances['dataset_instances']), opt.batch_size, shuffle=False,
                                      collate_fn=my_collate)
            dev_indomain_data_loaders.append(dev_loader)

        dev_outdomain_data_loaders = []
        for dev_dataset_instances in dev_outdomain_datasets_instances:
            dev_loader = DataLoader(MyDataset(dev_dataset_instances['dataset_instances']), opt.batch_size, shuffle=False,
                                      collate_fn=my_collate)
            dev_outdomain_data_loaders.append(dev_loader)


        logging.info("start training ...")

        best_test = -10
        bad_counter = 0
        for idx in range(opt.iter):
            epoch_start = time.time()

            model.train()

            logging.info("epoch: {} training start".format(idx))
            sum_loss = 0
            correct, total = 0, 0

            for train_idx, train_dataset in enumerate(train_datasets):
                logging.debug("train on {}".format(train_dataset['name']))

                train_loader = train_data_loaders[train_idx]
                train_iter = iter(train_loader)
                num_iter = len(train_loader)

                for i in range(num_iter):
                    input_ids, mask, segments, start_position, end_position = next(train_iter)

                    start_logits, end_logits = model.forward(input_ids, segments, mask)

                    loss, total_this_batch, correct_this_batch = model.loss(start_logits, end_logits, start_position, end_position)

                    sum_loss += loss.item()

                    loss.backward()
                    optimizer.step()
                    model.zero_grad()

                    total += total_this_batch
                    correct += correct_this_batch

            epoch_finish = time.time()
            accuracy = 100.0 * correct / total
            logging.info("epoch: %s training finished. Time: %.2fs. loss: %.4f Accuracy %.2f" % (
                idx, epoch_finish - epoch_start, sum_loss / num_iter, accuracy))

            logging.info("##### indomain evaluation begin #####")
            indomain_macro_exact_match, indomain_macro_f1, indomain_all_pred_answers = evaluate(dev_indomain_datasets, dev_indomain_datasets_instances,
                                                                     dev_indomain_data_loaders, model, 'dev')
            logging.info("macro_exact_match %.4f, macro_f1 %.4f" % (indomain_macro_exact_match, indomain_macro_f1))
            logging.info("##### indomain evaluation end #####")

            logging.info("##### outdomain evaluation begin #####")
            outdomain_macro_exact_match, outdomain_macro_f1, outdomain_all_pred_answers = evaluate(dev_outdomain_datasets, dev_outdomain_datasets_instances,
                                                                     dev_outdomain_data_loaders, model, 'dev')
            logging.info("macro_exact_match %.4f, macro_f1 %.4f" % (outdomain_macro_exact_match, outdomain_macro_f1))
            logging.info("##### outdomain evaluation end #####")

            # Should we use indomain or outdomain performance as the final evaluation metrics? This should be discussed.
            if outdomain_macro_f1 > best_test:
                logging.info("Exceed previous best performance: %.4f" % (best_test))
                best_test = outdomain_macro_f1
                bad_counter = 0

                torch.save(model.state_dict(), os.path.join(opt.save, 'model.pth'))
                dump_results(dev_indomain_datasets, indomain_all_pred_answers, opt.save)
                dump_results(dev_outdomain_datasets, outdomain_all_pred_answers, opt.save)
            else:
                bad_counter += 1

            if bad_counter >= opt.patience:
                logging.info('Early Stop!')
                break


    elif opt.whattodo == 'test':

        makedir_and_clear(opt.predict)

        logging.info('loading test data...')
        test_datasets = []
        for test_jsonl_file in opt.test:
            test_dataset = load_data(test_jsonl_file, 'test')
            test_datasets.append(test_dataset)

        wp_tokenizer = BertTokenizer.from_pretrained(opt.bert_dir, do_lower_case=opt.do_lower_case)

        test_datasets_instances = []
        for test_dataset in test_datasets:
            test_dataset_instances = prepare_instance(test_dataset, opt, wp_tokenizer, 'test')
            test_datasets_instances.append(test_dataset_instances)

        if opt.gpu >= 0 and torch.cuda.is_available():
            device = torch.device("cuda", opt.gpu)
        else:
            device = torch.device("cpu")
        logging.info("use device {}".format(device))

        logging.info("loading model from {} ...".format(os.path.join(opt.save, 'model.pth')))
        model = BertForQuestionAnswering.from_pretrained(opt.bert_dir)
        model.load_state_dict(torch.load(os.path.join(opt.save, 'model.pth')))
        model.to(device)

        test_data_loaders = []
        for test_dataset_instances in test_datasets_instances:
            test_loader = DataLoader(MyDataset(test_dataset_instances['dataset_instances']), opt.batch_size, shuffle=False, collate_fn=my_collate)
            test_data_loaders.append(test_loader)

        logging.info("start test ...")
        _, _, all_pred_answers = evaluate(test_datasets, test_datasets_instances, test_data_loaders, model, 'test')
        dump_results(test_datasets, all_pred_answers, opt.predict)

    logging.info("end ......")






