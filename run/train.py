# -*- coding: UTF-8 -*-
from __future__ import division
import sys, time, os, cPickle
sys.path.append('..')
import dynet as dy
import numpy as np
import models
from lib import Vocab, DataLoader
from test import test
from config import Configurable

# from baseline.progress import create_progress_bar
# from baseline.utils import (
#     listify,
#     get_model_file,
#     get_metric_cmp,
#     convert_seq2seq_golds,
#     convert_seq2seq_preds
# )
# from baseline.train import (
#     Trainer,
#     create_trainer,
#     register_trainer,
#     register_training_func
# )
# from baseline.dy.optz import *
# from baseline.dy.dynety import *


import argparse
if __name__ == "__main__":
	np.random.seed(666)
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--config_file', default='./config.cfg')
	argparser.add_argument('--model', default='BaseParser')
	args, extra_args = argparser.parse_known_args()
	config = Configurable(args.config_file, extra_args)
	Parser = getattr(models, args.model)

	vocab = Vocab(config.train_file, config.pretrained_embeddings_file, config.min_occur_count)
	cPickle.dump(vocab, open(config.save_vocab_path, 'w'))
	parser = Parser(vocab, config.word_dims, config.pret_dims, config.lemma_dims, config.flag_dims, config.tag_dims, config.dropout_emb, 
					config.encoder_type, config.use_si_dropout,
					config.lstm_layers, config.lstm_hiddens, config.dropout_lstm_input, config.dropout_lstm_hidden, config.mlp_rel_size, config.dropout_mlp, 
					config.transformer_layers, config.transformer_heads, config.transformer_hiddens, config.transformer_ffn, config.transformer_dropout, config.transformer_maxlen, config.transformer_max_timescale,
					config.use_lm, config.lm_path, config.lm_dims, config.lm_hidden_size, config.lm_sentences,
					config.use_pos, config.use_lemma,
					config.unified)
	data_loader = DataLoader(config.train_file, config.num_buckets_train, vocab)
	pc = parser.parameter_collection
	trainer = dy.AdamTrainer(pc, config.learning_rate , config.beta_1, config.beta_2, config.epsilon)
	# optimizer = OptimizerManager(parser, lr_scheduler_type=config.lr_scheduler_type, optim=config.optim, warmup_steps=config.warmup_steps, 
	# 							eta=config.eta, patience=config.patience, clip=config.clip,
	# 							lr = config.learning_rate, beta1=config.beta_1, beta2 = config.beta_2, epsilon=config.epsilon)
	
	global_step = 0
	def update_parameters():
		trainer.learning_rate =config.learning_rate*config.decay**(global_step / config.decay_steps)
		trainer.update()

	epoch = 0
	best_F1 = 0.
	history = lambda x, y : open(os.path.join(config.save_dir, 'valid_history'),'a').write('%.2f %.2f\n'%(x,y))
	while global_step < config.train_iters:
		print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '\nStart training epoch #%d'%(epoch, )
		epoch += 1
		for words, lemmas, tags, arcs, rels, syn_masks, seq_lens in \
				data_loader.get_batches(batch_size = config.train_batch_size, shuffle = True):
			num = int(words.shape[1]/2)
			words_ = [words[:,:num], words[:,num:]]
			lemmas_ = [lemmas[:,:num], lemmas[:,num:]]
			tags_ = [tags[:,:num], tags[:,num:]]
			arcs_ = [arcs[:,:num], arcs[:,num:]]
			rels_ = [rels[:,:num], rels[:,num:]]
			syn_masks_ = [syn_masks[:,:num], syn_masks[:,num:]]
			seq_lens_ = [seq_lens[:num], seq_lens[num:]]
			for step in xrange(2):
				dy.renew_cg()
				rel_accuracy, loss = parser.run(words_[step], lemmas_[step], tags_[step], arcs_[step], rels_[step], syn_mask = syn_masks_[step], seq_lens = seq_lens_[step])
				loss = loss * 0.5
				loss_value = loss.scalar_value()
				loss.backward()
				sys.stdout.write("Step #%d: Acc: rel %.2f, loss %.3f\r\r" % 
									(global_step, rel_accuracy, loss_value))
				sys.stdout.flush()
			update_parameters()
			# optimizer.update()

			global_step += 1
			if global_step % config.validate_every == 0:
				print '\nTest on development set'
				dev_NF1, dev_F1 = test(parser, vocab, config.num_buckets_valid, config.test_batch_size, config.pro_dev_file, 
								config.raw_dev_file, os.path.join(config.save_dir, 'dev_valid_tmp'), 
								config.unified, config.dev_disambiguation_file, config.dev_disambiguation_accuracy)
				print '\nTest on test set'
				test_NF1, test_F1 = test(parser, vocab, config.num_buckets_valid, config.test_batch_size, config.pro_test_file, 
								config.raw_test_file, os.path.join(config.save_dir, 'test_valid_tmp'), 
								config.unified, config.test_disambiguation_file, config.test_disambiguation_accuracy)
				history(test_NF1, test_F1)
				if test_F1 > best_F1:
					best_F1 = test_F1
					os.system('cp %s.eval %s.eval.best' % (os.path.join(config.save_dir, 'dev_valid_tmp'), os.path.join(config.save_dir, 'dev_valid_tmp')))
					os.system('cp %s.eval %s.eval.best' % (os.path.join(config.save_dir, 'test_valid_tmp'), os.path.join(config.save_dir, 'test_valid_tmp')))
				if global_step > config.save_after and test_F1 > best_F1:
					parser.save(config.save_model_path)

				print '\tCurrent best:%.4f'% (best_F1*100)

