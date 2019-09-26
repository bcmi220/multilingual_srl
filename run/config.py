from ConfigParser import SafeConfigParser
import sys, os
sys.path.append('..')
import models

class Configurable(object):
	def __init__(self, config_file, extra_args):
		config = SafeConfigParser()
		config.read(config_file)
		if extra_args:
			extra_args = dict([ (k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
		for section in config.sections():
			for k, v in config.items(section):
				if k in extra_args:
					v = type(v)(extra_args[k])
					config.set(section, k, v)
		self._config = config
		if not os.path.isdir(self.save_dir):
			os.mkdir(self.save_dir)
		config.write(open(self.config_file,'w'))
		print 'Loaded config file sucessfully.'
		for section in config.sections():
			for k, v in config.items(section):
				print k, v

	@property
	def pretrained_embeddings_file(self):
		return self._config.get('Data','pretrained_embeddings_file')
	@property
	def data_dir(self):
		return self._config.get('Data','data_dir')
	@property
	def train_file(self):
		return self._config.get('Data','train_file')	
	@property
	def pro_dev_file(self):
		return self._config.get('Data','pro_dev_file')
	@property
	def raw_dev_file(self):
		return self._config.get('Data','raw_dev_file')
	@property
	def pro_test_file(self):
		return self._config.get('Data','pro_test_file')
	@property
	def raw_test_file(self):
		return self._config.get('Data','raw_test_file')
	@property
	def min_occur_count(self):
		return self._config.getint('Data','min_occur_count')
	@property
	def dev_disambiguation_accuracy(self):
		return self._config.getfloat('Data','dev_disambiguation_accuracy')
	@property
	def test_disambiguation_accuracy(self):
		return self._config.getfloat('Data','test_disambiguation_accuracy')
	@property
	def dev_disambiguation_file(self):
		return self._config.get('Data','dev_disambiguation_file')
	@property
	def test_disambiguation_file(self):
		return self._config.get('Data','test_disambiguation_file')

	@property
	def use_lm(self):
		return self._config.getboolean('Data','use_lm')
	@property
	def use_pos(self):
		return self._config.getboolean('Data','use_pos')
	@property
	def use_lemma(self):
		return self._config.getboolean('Data','use_lemma')
	@property
	def lm_path(self):
		return self._config.get('Data','lm_path')
	@property
	def lm_dims(self):
		return self._config.getint('Data','lm_dims')
	@property
	def lm_hidden_size(self):
		return self._config.getint('Data','lm_hidden_size')
	@property
	def lm_sentences(self):
		return self._config.get('Data','lm_sentences')
	@property
	def unified(self):
		return self._config.getboolean('Data','unified')
	
	@property
	def save_dir(self):
		return self._config.get('Save','save_dir')
	@property
	def config_file(self):
		return self._config.get('Save','config_file')
	@property
	def save_model_path(self):
		return self._config.get('Save','save_model_path')
	@property
	def save_vocab_path(self):
		return self._config.get('Save','save_vocab_path')
	@property
	def load_dir(self):
		return self._config.get('Save','load_dir')
	@property
	def load_model_path(self):
		return self._config.get('Save', 'load_model_path')
	@property
	def load_vocab_path(self):
		return self._config.get('Save', 'load_vocab_path')

	@property
	def encoder_type(self):
		return self._config.get('Network','encoder_type')
	@property
	def use_si_dropout(self):
		return self._config.getboolean('Network','use_si_dropout')
	
	@property
	def lstm_layers(self):
		return self._config.getint('Network','lstm_layers')
	@property
	def word_dims(self):
		return self._config.getint('Network','word_dims')
	@property
	def pret_dims(self):
		return self._config.getint('Network','pretrain_dims')
	@property
	def lemma_dims(self):
		return self._config.getint('Network','lemma_dims')
	@property
	def flag_dims(self):
		return self._config.getint('Network','head_flag_dims')
	@property
	def tag_dims(self):
		return self._config.getint('Network','tag_dims')
	@property
	def dropout_emb(self):
		return self._config.getfloat('Network','dropout_emb')
	@property
	def lstm_hiddens(self):
		return self._config.getint('Network','lstm_hiddens')
	@property
	def dropout_lstm_input(self):
		return self._config.getfloat('Network','dropout_lstm_input')
	@property
	def dropout_lstm_hidden(self):
		return self._config.getfloat('Network','dropout_lstm_hidden')
	@property
	def mlp_arc_size(self):
		return self._config.getint('Network','mlp_arc_size')
	@property
	def mlp_rel_size(self):
		return self._config.getint('Network','mlp_rel_size')
	@property
	def dropout_mlp(self):
		return self._config.getfloat('Network','dropout_mlp')
	@property
	def transformer_layers(self):
		return self._config.getint('Network','transformer_layers')
	@property
	def transformer_heads(self):
		return self._config.getint('Network','transformer_heads')
	@property
	def transformer_hiddens(self):
		return self._config.getint('Network','transformer_hiddens')
	@property
	def transformer_ffn(self):
		return self._config.getint('Network','transformer_ffn')
	@property
	def transformer_dropout(self):
		return self._config.getfloat('Network','transformer_dropout')
	@property
	def transformer_maxlen(self):
		return self._config.getint('Network','transformer_maxlen')
	@property
	def transformer_max_timescale(self):
		return self._config.getfloat('Network','transformer_max_timescale')
	
	
	@property
	def optim(self):
		return self._config.get('Optimizer','optim')
	@property
	def lr_scheduler_type(self):
		return self._config.get('Optimizer','lr_scheduler_type')
	@property
	def warmup_steps(self):
		return self._config.getint('Optimizer','warmup_steps')
	@property
	def eta(self):
		return self._config.getfloat('Optimizer','eta')
	@property
	def patience(self):
		return self._config.getint('Optimizer','patience')
	@property
	def clip(self):
		return self._config.getfloat('Optimizer','clip')
	@property
	def learning_rate(self):
		return self._config.getfloat('Optimizer','learning_rate')
	@property
	def decay(self):
		return self._config.getfloat('Optimizer','decay')
	@property
	def decay_steps(self):
		return self._config.getint('Optimizer','decay_steps')
	@property
	def beta_1(self):
		return self._config.getfloat('Optimizer','beta_1')
	@property
	def beta_2(self):
		return self._config.getfloat('Optimizer','beta_2')
	@property
	def epsilon(self):
		return self._config.getfloat('Optimizer','epsilon')

	@property
	def num_buckets_train(self):
		return self._config.getint('Run','num_buckets_train')
	@property
	def num_buckets_valid(self):
		return self._config.getint('Run','num_buckets_valid')
	@property
	def num_buckets_test(self):
		return self._config.getint('Run','num_buckets_test')
	@property	
	def train_iters(self):
		return self._config.getint('Run','train_iters')
	@property	
	def train_batch_size(self):
		return self._config.getint('Run','train_batch_size')
	@property
	def test_batch_size(self):
		return self._config.getint('Run','test_batch_size')
	@property	
	def validate_every(self):
		return self._config.getint('Run','validate_every')
	@property
	def save_after(self):
		return self._config.getint('Run','save_after')

import argparse
if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--config_file', default='../configs/default.cfg')
	argparser.add_argument('--model', default='BaseParser')
	args, extra_args = argparser.parse_known_args()

	config = Configurable(args.config_file, extra_args)
	Parser = getattr(models, args.model)	
