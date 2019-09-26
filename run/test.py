#Embedded file name: /home/dengcai/code/run/test.py
from __future__ import division
import sys
sys.path.append('..')
import time, os, cPickle
import dynet as dy
import models
from lib import Vocab, DataLoader
from config import Configurable
import random
import copy

# gold_rels = None
# test_data = None
# test_stat = None


def test(parser, vocab, num_buckets_test, test_batch_size, pro_test_file, raw_test_file, output_file,
	unified = True, disambiguation_file = None, disambiguation_accuracy = None):
	if not unified:
		assert ((disambiguation_file is not None and len(disambiguation_file)>0) or disambiguation_accuracy is not None), \
				'The accuracy of predicate disambiguation shold be provied.'
	data_loader = DataLoader(pro_test_file, num_buckets_test, vocab)
	record = data_loader.idx_sequence
	results = [None] * len(record)
	idx = 0
	for words, lemmas, tags, arcs, rels, syn_masks, seq_lens in \
			data_loader.get_batches(batch_size = test_batch_size, shuffle = False):
		dy.renew_cg()
		outputs = parser.run(words, lemmas, tags, arcs, isTrain = False, syn_mask = syn_masks, seq_lens=seq_lens)
		for output in outputs:
			sent_idx = record[idx]
			results[sent_idx] = output
			idx += 1
	rels = reduce(lambda x, y: x + y, [ list(result) for result in results ])
	
	# global gold_rels
	# global test_data
	# global test_stat

	# if not gold_rels:
	gold_rels = []
	gold_sent = []
	with open(pro_test_file) as f:
		for line in f:
			info = line.strip().split()
			if info:
				assert len(info) == 11, 'Illegal line: %s' % line
				gold_sent.append(info[7])
			else:
				if len(gold_sent)>0:
					gold_rels.append(gold_sent)
					gold_sent = []
		if len(gold_sent)>0:
			gold_rels.append(gold_sent)

	
	# if not test_data:
	#print 'prepare for writing out the prediction'
	with open(raw_test_file) as f:
		test_data = []
		test_sent = []
		test_stat = []
		pred_index = []
		for line in f:
			info = line.strip().split()
			if info:
				test_sent.append([info[0], info[1], info[2], info[3], info[4], info[5], 
								info[6], info[7], info[8], info[9], info[10], info[11], 
								info[12], '_' if unified else info[13]])
				if info[12] == 'Y':
					pred_index.append(int(info[0]))
			elif len(test_sent) > 0:
				test_data.append(test_sent)
				test_sent = []
				test_stat.append(pred_index)
				pred_index = []
		if len(test_sent) > 0:
			test_data.append(test_sent)
			test_sent = []
			test_stat.append(pred_index)
			pred_index = []


	if disambiguation_file is not None and len(disambiguation_file)>0:
		gold_disamb = []
		gold_dsent = []
		with open(disambiguation_file) as f:
			for line in f:
				info = line.strip().split()
				if info:
					gold_dsent.append(info[0])
				else:
					if len(gold_dsent)>0:
						gold_disamb.append(gold_dsent)
						gold_dsent = []
			if len(gold_dsent)>0:
				gold_disamb.append(gold_dsent)

		assert len(test_data) == len(gold_disamb)

	idx = 0
	oidx = 0
	with open(output_file, 'w') as f:
		for test_s, pred_index in zip(test_data, test_stat):
			test_sent = copy.deepcopy(test_s)
			for p_idx in pred_index:
				if unified:
					test_sent[p_idx - 1][13] = test_sent[p_idx - 1][2] + '.' + vocab.id2rel(results[idx][0])
				else:
					if disambiguation_file is not None and len(disambiguation_file)>0:
						test_sent[p_idx - 1][13] = gold_disamb[oidx][p_idx - 1]
					else:
						if random.random() < disambiguation_accuracy:
							test_sent[p_idx - 1][13] = test_sent[p_idx - 1][13]
						else:
							test_sent[p_idx - 1][13] = test_sent[p_idx - 1][2]
				if unified:
					for i, p_rel in enumerate(results[idx][1:]):	
						test_sent[i].append(vocab.id2rel(p_rel))
				else:
					for i, p_rel in enumerate(results[idx]):	
						test_sent[i].append(vocab.id2rel(p_rel))
				idx += 1
			oidx += 1
			for tokens in test_sent:
				f.write('\t'.join(tokens))
				f.write('\n')
			f.write('\n')
	
	predict_args = 0.
	correct_args = 0.
	gold_args = 0.
	correct_preds = 0.
	gold_preds = len(gold_rels)
	num_correct = 0.
	total = 0.
	idx = 0
	for sent in gold_rels:
		
		for i in range(len(sent)):
			gold = sent[i]
			pred = rels[idx]
			if unified and i==0:
				if sent[i] == vocab.id2rel(pred):
					correct_preds += 1
			else:
				total += 1
				if vocab.id2rel(pred)!='_':
					predict_args += 1
				if gold != '_':
					gold_args += 1
				if gold != '_' and gold == vocab.id2rel(pred):
					correct_args += 1
				if gold == vocab.id2rel(pred):
					num_correct += 1
			idx += 1
	
	correct_preds = correct_preds if unified else gold_preds * disambiguation_accuracy
	P = (correct_args + correct_preds) / (predict_args + gold_preds + 1e-13)
	R = (correct_args + correct_preds) / (gold_args + gold_preds + 1e-13)
	NP = correct_args / (predict_args + 1e-13)
	NR = correct_args / (gold_args + 1e-13)	   
	F1 = 2 * P * R / (P + R + 1e-13)
	NF1 = 2 * NP * NR / (NP + NR + 1e-13)
	print '\teval accurate:%.4f predict:%d golden:%d correct:%d' % \
				(num_correct / total * 100, predict_args, gold_args, correct_args)
	print '\tP:%.4f R:%.4f F1:%.4f' % (P * 100, R * 100, F1 * 100)
	print '\tNP:%.2f NR:%.2f NF1:%.2f' % (NP * 100, NR * 100, NF1 * 100)
	print '\tpredicate disambiguation accurate:%.2f' % (correct_preds / gold_preds * 100)
	os.system('perl ../lib/eval.pl -g %s -s %s > %s.eval' % (raw_test_file, output_file, output_file))
	return NF1, F1


import argparse
if __name__ == '__main__':
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--config_file', default='../configs/default.cfg')
	argparser.add_argument('--model', default='BaseParser')
	argparser.add_argument('--output_file', default='test.predict')
	args, extra_args = argparser.parse_known_args()
	config = Configurable(args.config_file, extra_args)
	Parser = getattr(models, args.model)
	vocab = cPickle.load(open(config.load_vocab_path))
	parser = Parser(vocab, config.word_dims, config.pret_dims, config.lemma_dims, config.flag_dims, config.tag_dims, config.dropout_emb, 
					config.encoder_type, config.use_si_dropout,
					config.lstm_layers, config.lstm_hiddens, config.dropout_lstm_input, config.dropout_lstm_hidden, config.mlp_rel_size, config.dropout_mlp, 
					config.transformer_layers, config.transformer_heads, config.transformer_hiddens, config.transformer_ffn, config.transformer_dropout, config.transformer_maxlen, config.transformer_max_timescale,
					config.use_lm, config.lm_path, config.lm_dims, config.lm_hidden_size, config.lm_sentences, 
					config.unified)
	parser.load(config.load_model_path)
	test(parser, vocab, config.num_buckets_test, config.test_batch_size, config.pro_test_file, 
		config.raw_test_file, args.output_file, config.unified, config.test_disambiguation_file, config.test_disambiguation_accuracy)

