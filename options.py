import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-whattodo', type=str, default="train", choices=["train","test"])
parser.add_argument('-train', type=str, default='', help="list of train data path, splitted by comma, e.g., /data_path1,/data_path2,...")
parser.add_argument('-train_sample_size', type=int, default=75000)
parser.add_argument('-dev_indomain', type=str, default='', help="the same as -train")
parser.add_argument('-dev_outdomain', type=str, default='', help="the same as -train")
parser.add_argument('-dev_sample_size', type=int, default=1000)
parser.add_argument('-test', type=str, default='', help="the same as -train")


parser.add_argument('-verbose', action='store_true', default=False, help="output debug info if true")
parser.add_argument('-random_seed', type=int, default=1, help="random initial seed if 0")
parser.add_argument("-max_query_length", default=64, type=int, help="The maximum number of tokens for the question. "
                     "Questions longer than this will be truncated to this length.")
parser.add_argument('-max_seq_length', type=int, default=512, help="The maximum total input sequence (question+context)"
                                                                   " length after WordPiece tokenization. ")
parser.add_argument("-max_answer_length", default=30, type=int, help="The maximum length of an answer that can be generated.")
parser.add_argument('-skip_no_answer', action='store_true', default=False, help="see mrqa_official_eval.py")


parser.add_argument('-bert_dir', type=str, help="bert directory that includes bert_config, vocab and model")
parser.add_argument('-do_lower_case', action='store_true', default=False, help="used with BertTokenizer")
parser.add_argument('-save', type=str, default='./save', help="output directory for training")
parser.add_argument('-predict', type=str, default='./predict', help="output directory for test")

parser.add_argument('-batch_size', type=int, default=16)
parser.add_argument('-lr', type=float, default=0.00001, help="learning rate")
parser.add_argument('-l2', type=float, default=1e-8, help="l2 regularization")
parser.add_argument('-iter', type=int, default=100, help="max iteration")
parser.add_argument('-gpu', type=int, default=-1, help="if gpu<0, use cpu, otherwise use the specific gpu")
parser.add_argument('-patience', type=int, default=20, help="if the result doesn't rise after several iteration, training will stop")
parser.add_argument('-optim', type=str, default="adam", choices=["adam","bert_adam"])
parser.add_argument("-warmup_proportion", default=0.1, type=float)
parser.add_argument('-gradient_accumulation_steps', type=int, default=1)

opt = parser.parse_args()

opt.train = opt.train.strip().split(',')
opt.dev_indomain = opt.dev_indomain.strip().split(',')
opt.dev_outdomain = opt.dev_outdomain.strip().split(',')
opt.test = opt.test.strip().split(',')
