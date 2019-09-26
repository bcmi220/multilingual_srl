# -*- coding: UTF-8 -*-
from __future__ import division
import dynet as dy
import numpy as np
from lib import *
from baseline.dy.seq2seq import TransformerEncoderWrapper
import h5py
import math

class BaseParser(object):
	def __init__(self, vocab, word_dims, pret_dims, lemma_dims, flag_dims, tag_dims, dropout_dim,
				encoder_type, use_si_droput,
				lstm_layers, lstm_hiddens, dropout_lstm_input, dropout_lstm_hidden, mlp_size, dropout_mlp, 
				transformer_layers, transformer_heads, transformer_hiddens, transformer_ffn, transformer_dropout, transformer_maxlen, transformer_max_timescale,
				use_lm, lm_path, lm_dims, lm_hidden_size, lm_sentences, 
				use_pos, use_lemma,
				unified = True): #, use_biaffine=True
		
		pc = dy.ParameterCollection()
		self._vocab = vocab
		self.word_embs = pc.lookup_parameters_from_numpy(vocab.get_word_embs(word_dims))
		self.pret_word_embs = pc.lookup_parameters_from_numpy(vocab.get_pret_embs(pret_dims))
		self.flag_embs = pc.lookup_parameters_from_numpy(vocab.get_flag_embs(flag_dims))
		self.use_lemma = use_lemma
		if self.use_lemma:
			self.lemma_embs = pc.lookup_parameters_from_numpy(vocab.get_lemma_embs(lemma_dims))
		self.use_pos = use_pos
		if self.use_pos:
			self.tag_embs = pc.lookup_parameters_from_numpy(vocab.get_tag_embs(tag_dims))
		
		self.use_lm = use_lm
		self.lm_dims = lm_dims
		self.lm_hidden_size = lm_hidden_size
		if self.use_lm:
			self.lm_hidden_W = pc.add_parameters((self.lm_dims, self.lm_hidden_size))
			self.lm_hidden_b = pc.add_parameters((self.lm_hidden_size,), init = dy.ConstInitializer(0.))

		self.use_si_droput = use_si_droput

		
		input_dims = word_dims + pret_dims  + flag_dims
		if self.use_pos:
			input_dims += tag_dims
		if self.use_lemma:
			input_dims += lemma_dims
		if self.use_lm:
			input_dims += lm_hidden_size
		
		self.encoder_type = encoder_type
		if self.encoder_type == 'rnn':
			self.LSTM_builders = []
			f = orthonormal_VanillaLSTMBuilder(1, input_dims, lstm_hiddens, pc)
			b = orthonormal_VanillaLSTMBuilder(1, input_dims, lstm_hiddens, pc)
			self.LSTM_builders.append((f, b))
			for i in xrange(lstm_layers - 1):
				f = orthonormal_VanillaLSTMBuilder(1, 2 * lstm_hiddens, lstm_hiddens, pc)
				b = orthonormal_VanillaLSTMBuilder(1, 2 * lstm_hiddens, lstm_hiddens, pc)
				self.LSTM_builders.append((f, b))
			self.dropout_lstm_input = dropout_lstm_input
			self.dropout_lstm_hidden = dropout_lstm_hidden

			context_size = 2 * lstm_hiddens
		else: # for transformer
			self.input_dims = input_dims
			max_len = transformer_maxlen
			max_timescale = transformer_max_timescale
			log_timescale_inc = math.log(transformer_max_timescale) / input_dims
			inv_timescale = np.exp(np.arange(0, input_dims, 2) * -log_timescale_inc)
			pe = np.zeros((max_len, input_dims))
			position = np.expand_dims(np.arange(max_len), 1)
			pe[:, 0::2] = np.sin(position * inv_timescale)
			pe[:, 1::2] = np.cos(position * inv_timescale)
			self.pe = pe

			self.transformer = TransformerEncoderWrapper(input_dims, pc, transformer_hiddens, transformer_ffn, transformer_heads, transformer_layers, transformer_dropout)

			context_size = transformer_hiddens
		
		W = orthonormal_initializer(mlp_size, context_size)
		self.mlp_arg_W = pc.parameters_from_numpy(W)
		self.mlp_pred_W = pc.parameters_from_numpy(W)
		self.mlp_arg_b = pc.add_parameters((mlp_size,), init = dy.ConstInitializer(0.))
		self.mlp_pred_b = pc.add_parameters((mlp_size,), init = dy.ConstInitializer(0.))
		self.mlp_size = mlp_size
		self.dropout_mlp = dropout_mlp

		#self.use_biaffine = use_biaffine
		self.rel_W = pc.add_parameters((vocab.rel_size * (mlp_size +1) , mlp_size + 1), 
		 									init = dy.ConstInitializer(0.))
		# if use_biaffine:
		# 	self.rel_W = pc.add_parameters((vocab.rel_size * (mlp_size +1) , mlp_size + 1), 
		# 									init = dy.ConstInitializer(0.))
		# else: # use FFN scorer
		# 	self.ffn_hidden_size = 150
		# 	self.ffn_layer1_W = pc.add_parameters((self.ffn_hidden_size, mlp_size*2),init = dy.ConstInitializer(0.))
		# 	self.ffn_layer1_b = pc.add_parameters((self.ffn_hidden_size,), init = dy.ConstInitializer(0.))
		# 	self.ffn_layer2_W = pc.add_parameters((self.ffn_hidden_size, self.ffn_hidden_size),init = dy.ConstInitializer(0.))
		# 	self.ffn_layer2_b = pc.add_parameters((self.ffn_hidden_size,), init = dy.ConstInitializer(0.))
		# 	self.rel_W = pc.add_parameters((vocab.rel_size, self.ffn_hidden_size),init = dy.ConstInitializer(0.))
		

		self._unified = unified
		self.pc = pc

		self.lm_data = h5py.File(lm_path, 'r')
		self.lm_sentences = []
		with open(lm_sentences, 'r') as f:
			for line in f.readlines():
				if len(line.strip()) > 0:
					self.lm_sentences.append(line.strip())
		self.lm_sentences = [[item.lower() for item in line.split()] for line in list(self.lm_sentences)]
		self.lm_dict = {}

		def _emb_mask_generator(seq_len, batch_size):
			ret = []
			for i in xrange(seq_len):
				word_mask = np.random.binomial(1, 1. - dropout_dim, batch_size).astype(np.float32)
				tag_mask = np.random.binomial(1, 1. - dropout_dim, batch_size).astype(np.float32)
				scale = 3. / (2.*word_mask + tag_mask + 1e-12)
				word_mask *= scale
				tag_mask *= scale
				word_mask = dy.inputTensor(word_mask, batched = True)
				tag_mask = dy.inputTensor(tag_mask, batched = True)
				ret.append((word_mask, tag_mask))
			return ret
		self.generate_emb_mask = _emb_mask_generator


	@property 
	def parameter_collection(self):
		return self.pc


	def run(self, word_inputs, lemma_inputs, tag_inputs, pred_golds, rel_targets = None, isTrain = True, syn_mask = None, seq_lens = None):
		# inputs, targets: seq_len x batch_size
		def dynet_flatten_numpy(ndarray):
			return np.reshape(ndarray, (-1,), 'F')

		batch_size = word_inputs.shape[1]
		seq_len = word_inputs.shape[0]
		mask = np.greater(word_inputs, self._vocab.PAD).astype(np.float32)
		num_tokens = int(np.sum(mask))

		word_embs = [dy.lookup_batch(self.word_embs, 
									np.where(w < self._vocab.words_in_train, w, self._vocab.UNK)
						) for w in word_inputs]

		if self.use_lm:
			lm_embs = np.zeros((batch_size, seq_len, self.lm_dims),dtype=float)
			for idx in range(batch_size):
				if self._unified:
					txt = [self._vocab.id2word(w) for w in word_inputs[1:,idx] if self._vocab.id2word(w) != '<PAD>']
					key = ' '.join(txt)
					key = self.lm_dict.get(key, None)
					if key is None:
						for sidx in range(len(self.lm_sentences)):
							line = self.lm_sentences[sidx]
							if len(line) != len(txt):
								continue
							found = True
							for mdx in range(len(line)):
								if line[mdx] != txt[mdx] and txt[mdx] != '<UNK>':
									found = False
									break
							if found:
								key = str(sidx)
								self.lm_dict[' '.join(txt)] = key
								break
					assert key is not None
					lm_embs[idx, 1:1+len(txt),:] = self.lm_data[key][...]
				else:
					txt = [self._vocab.id2word(w) for w in word_inputs[:,idx] if self._vocab.id2word(w) != '<PAD>']
					key = ' '.join(txt)
					key = self.lm_dict.get(key, None)
					if key is None:
						for sidx in range(len(self.lm_sentences)):
							line = self.lm_sentences[sidx]
							if len(line) != len(txt):
								continue
							found = True
							for mdx in range(len(line)):
								if line[mdx] != txt[mdx] and txt[mdx] != '<UNK>':
									found = False
									break
							if found:
								key = str(sidx)
								self.lm_dict[' '.join(txt)] = key
								break
					assert key is not None
					lm_embs[idx, :len(txt),:] = self.lm_data[key][...]
			lm_embs = lm_embs.transpose(1,2,0)
			lm_embs = [dy.inputTensor(e, batched=True) for e in list(lm_embs)]

		pre_embs = [dy.lookup_batch(self.pret_word_embs, w) for w in word_inputs]
		flag_embs = [dy.lookup_batch(self.flag_embs, 
									np.array(w == i + 1, dtype=np.int)
						) for i, w in enumerate(pred_golds)]
		if self.use_lemma:
			lemma_embs = [dy.lookup_batch(self.lemma_embs, lemma) for lemma in lemma_inputs]
		if self.use_pos:
			tag_embs = [dy.lookup_batch(self.tag_embs, pos) for pos in tag_inputs]
		
		if self.use_lm:
			if isTrain:
				emb_masks = self.generate_emb_mask(seq_len, batch_size)
				if self.use_lemma and self.use_pos:
					emb_inputs = [dy.concatenate([dy.cmult(word, wm), dy.cmult(pre, wm), dy.cmult(flag, wm), 
													dy.cmult(lemma, wm), dy.cmult(lme, wm), dy.cmult(pos, posm)]) 
									for word, pre, flag, lemma, pos, lme, (wm, posm) in 
										zip(word_embs, pre_embs, flag_embs, lemma_embs, tag_embs, lm_embs, emb_masks)]
				elif self.use_lemma:
					emb_inputs = [dy.concatenate([dy.cmult(word, wm), dy.cmult(pre, wm), dy.cmult(flag, wm), 
													dy.cmult(lemma, wm), dy.cmult(lme, wm)]) 
									for word, pre, flag, lemma, pos, lme, (wm, posm) in 
										zip(word_embs, pre_embs, flag_embs, lemma_embs, lm_embs, emb_masks)]
				elif self.use_pos:
					emb_inputs = [dy.concatenate([dy.cmult(word, wm), dy.cmult(pre, wm), dy.cmult(flag, wm), 
													dy.cmult(lme, wm), dy.cmult(pos, posm)]) 
									for word, pre, flag, pos, lme, (wm, posm) in 
										zip(word_embs, pre_embs, flag_embs, tag_embs, lm_embs, emb_masks)]
				else:
					emb_inputs = [dy.concatenate([dy.cmult(word, wm), dy.cmult(pre, wm), dy.cmult(flag, wm), 
													dy.cmult(lme, wm)]) 
									for word, pre, flag, lme, (wm, posm) in 
										zip(word_embs, pre_embs, flag_embs, lm_embs, emb_masks)]
				
			else:
				if self.use_lemma and self.use_pos:
					emb_inputs = [dy.concatenate([word, pre, flag, lemma, lme, pos]) 
									for word, pre, flag, lemma, lme, pos in 
										zip(word_embs, pre_embs, flag_embs, lemma_embs, lm_embs, tag_embs)]
				elif self.use_lemma:
					emb_inputs = [dy.concatenate([word, pre, flag, lme, pos]) 
									for word, pre, flag, lemma, lme, pos in 
										zip(word_embs, pre_embs, flag_embs, lm_embs, tag_embs)]
				elif self.use_pos:
					emb_inputs = [dy.concatenate([word, pre, flag, lemma, lme]) 
									for word, pre, flag, lemma, lme in 
										zip(word_embs, pre_embs, flag_embs, lemma_embs, lm_embs)]
				else:
					emb_inputs = [dy.concatenate([word, pre, flag, lme]) 
									for word, pre, flag, lme in 
										zip(word_embs, pre_embs, flag_embs, lm_embs)]
		else:
			if isTrain:
				emb_masks = self.generate_emb_mask(seq_len, batch_size)
				if self.use_lemma and self.use_pos:
					emb_inputs = [dy.concatenate([dy.cmult(word, wm), dy.cmult(pre, wm), dy.cmult(flag, wm), 
													dy.cmult(lemma, wm), dy.cmult(pos, posm)]) 
									for word, pre, flag, lemma, pos, (wm, posm) in 
										zip(word_embs, pre_embs, flag_embs, lemma_embs, tag_embs, emb_masks)]
				elif self.use_lemma:
					emb_inputs = [dy.concatenate([dy.cmult(word, wm), dy.cmult(pre, wm), dy.cmult(flag, wm), 
													dy.cmult(lemma, wm)]) 
									for word, pre, flag, lemma, (wm, posm) in 
										zip(word_embs, pre_embs, flag_embs, lemma_embs, emb_masks)]
				elif self.use_pos:
					emb_inputs = [dy.concatenate([dy.cmult(word, wm), dy.cmult(pre, wm), dy.cmult(flag, wm), 
													 dy.cmult(pos, posm)]) 
									for word, pre, flag, pos, (wm, posm) in 
										zip(word_embs, pre_embs, flag_embs, tag_embs, emb_masks)]
				else:
					emb_inputs = [dy.concatenate([dy.cmult(word, wm), dy.cmult(pre, wm), dy.cmult(flag, wm)]) 
									for word, pre, flag, (wm, posm) in 
										zip(word_embs, pre_embs, flag_embs, emb_masks)]
				
			else:
				if self.use_lemma and self.use_pos:
					emb_inputs = [dy.concatenate([word, pre, flag, lemma, pos]) 
									for word, pre, flag, lemma, pos in 
										zip(word_embs, pre_embs, flag_embs, lemma_embs, tag_embs)]
				elif self.use_lemma:
					emb_inputs = [dy.concatenate([word, pre, flag, lemma]) 
									for word, pre, flag, lemma in 
										zip(word_embs, pre_embs, flag_embs, lemma_embs)]
				elif self.use_pos:
					emb_inputs = [dy.concatenate([word, pre, flag, pos]) 
									for word, pre, flag, pos in 
										zip(word_embs, pre_embs, flag_embs, tag_embs)]
				else:
					emb_inputs = [dy.concatenate([word, pre, flag]) 
									for word, pre, flag in 
										zip(word_embs, pre_embs, flag_embs)]


		if self.encoder_type == 'rnn':
			top_recur = dy.concatenate_cols(
							biLSTM(self.LSTM_builders, emb_inputs, batch_size, 
									self.dropout_lstm_input if isTrain else 0., 
									self.dropout_lstm_hidden if isTrain else 0.))
		else:

			emb_inputs = dy.concatenate_cols(emb_inputs)

			emb_inputs = emb_inputs * math.sqrt(self.input_dims)

			emb_inputs = emb_inputs + dy.transpose(dy.inputTensor(self.pe[:seq_len]))

			emb_inputs = dy.transpose(emb_inputs)

			encoder_outputs = self.transformer(emb_inputs, src_len=seq_lens, train=isTrain)

			top_recur = encoder_outputs.output

			top_recur = dy.concatenate_cols(top_recur)

			#print(top_recur.dim())

		if isTrain:
			top_recur = dy.dropout_dim(top_recur, 1, self.dropout_mlp)

		W_arg, b_arg = self.mlp_arg_W.expr(), self.mlp_arg_b.expr() #dy.parameter(self.mlp_arg_W), dy.parameter(self.mlp_arg_b)
		W_pred, b_pred = dy.parameter(self.mlp_pred_W), dy.parameter(self.mlp_pred_b)
		arg_hidden = leaky_relu(dy.affine_transform([b_arg, W_arg, top_recur]))
		# pred_hidden = leaky_relu(dy.affine_transform([b_pred, W_pred, top_recur]))
		predicates_1D = pred_golds[0]
		pred_recur = dy.pick_batch(top_recur, predicates_1D, dim=1)
		pred_hidden = leaky_relu(dy.affine_transform([b_pred, W_pred, pred_recur]))
		if isTrain:
			arg_hidden = dy.dropout_dim(arg_hidden, 1, self.dropout_mlp)
			# pred_hidden = dy.dropout_dim(pred_hidden, 1, self.dropout_mlp)
			pred_hidden = dy.dropout(pred_hidden, self.dropout_mlp)

		W_rel = dy.parameter(self.rel_W)

		# rel_logits = bilinear(arg_hidden, W_rel, pred_hidden, self.mlp_size, seq_len, batch_size, 
		# 						num_outputs = self._vocab.rel_size, bias_x = True, bias_y = True)
		# # (#pred x rel_size x #arg) x batch_size
		
		# flat_rel_logits = dy.reshape(rel_logits, (seq_len, self._vocab.rel_size), seq_len * batch_size)
		# # (#pred x rel_size) x (#arg x batch_size)

		# predicates_1D = dynet_flatten_numpy(pred_golds)
		# partial_rel_logits = dy.pick_batch(flat_rel_logits, predicates_1D)
		# # (rel_size) x (#arg x batch_size)

		if self.use_si_droput and syn_mask is not None:
			syn_mask = np.expand_dims(syn_mask, axis=0) # (1, seq_len, batch_size)
			arg_hidden = dy.cmult(arg_hidden, dy.inputTensor(syn_mask, batched=True))
		
		rel_logits = bilinear(arg_hidden, W_rel, pred_hidden, self.mlp_size, seq_len, 1, batch_size, 
		 							num_outputs = self._vocab.rel_size, bias_x = True, bias_y = True)
		# if self.use_biaffine:
		# 	rel_logits = bilinear(arg_hidden, W_rel, pred_hidden, self.mlp_size, seq_len, 1, batch_size, 
		# 							num_outputs = self._vocab.rel_size, bias_x = True, bias_y = True)
		# else:
		# 	pred_hidden = dy.reshape(pred_hidden, (self.mlp_size, 1), batch_size)
		# 	preds_hidden = [pred_hidden for _ in xrange(seq_len)]
		# 	preds_hidden = dy.concatenate(preds_hidden, d=1)
		# 	rel_hidden = dy.concatenate([preds_hidden, arg_hidden], d=0)  # (2*mlp_size x seq_len) x batch_size
		# 	flat_rel_hidden = dy.reshape(rel_hidden, (self.mlp_size*2, ), seq_len * batch_size)

		# 	W_ffn_layer1 = dy.parameter(self.ffn_layer1_W)
		# 	b_ffn_layer1 = dy.parameter(self.ffn_layer1_b)
		# 	W_ffn_layer2 = dy.parameter(self.ffn_layer2_W)
		# 	b_ffn_layer2 = dy.parameter(self.ffn_layer2_b)

		# 	flat_rel_hidden = leaky_relu(dy.affine_transform([b_ffn_layer1, W_ffn_layer1, flat_rel_hidden]))
		# 	flat_rel_hidden = leaky_relu(dy.affine_transform([b_ffn_layer2, W_ffn_layer2, flat_rel_hidden]))
		# 	flat_rel_hidden = W_rel * flat_rel_hidden
		# 	rel_logits = dy.reshape(flat_rel_hidden, (1, self._vocab.rel_size, seq_len), batch_size)

		# (1 x rel_size x #arg) x batch_size
		flat_rel_logits = dy.reshape(rel_logits, (1, self._vocab.rel_size), seq_len * batch_size)
		# (1 x rel_size) x (#arg x batch_size)

		predicates_1D = np.zeros(dynet_flatten_numpy(pred_golds).shape[0])
		partial_rel_logits = dy.pick_batch(flat_rel_logits, predicates_1D)
		# (1 x rel_size) x (#arg x batch_size)

		if isTrain:
			mask_1D = dynet_flatten_numpy(mask)
			mask_1D_tensor = dy.inputTensor(mask_1D, batched = True)
			rel_preds = partial_rel_logits.npvalue().argmax(0)
			targets_1D = dynet_flatten_numpy(rel_targets)
			rel_correct = np.equal(rel_preds, targets_1D).astype(np.float32) * mask_1D
			rel_accuracy = np.sum(rel_correct)/ num_tokens
			losses = dy.pickneglogsoftmax_batch(partial_rel_logits, targets_1D)
			rel_loss = dy.sum_batches(losses * mask_1D_tensor) / num_tokens
			return rel_accuracy, rel_loss

		# rel_probs = np.transpose(np.reshape(dy.softmax(dy.transpose(flat_rel_logits)).npvalue(), 
		# 									(self._vocab.rel_size, seq_len, seq_len, batch_size), 'F'))
		
		rel_probs = np.transpose(np.reshape(dy.softmax(dy.transpose(flat_rel_logits)).npvalue(), 
											(self._vocab.rel_size, 1, seq_len, batch_size), 'F'))
		outputs = []

		# for msk, pred_gold, rel_prob in zip(np.transpose(mask), pred_golds.T, rel_probs):
		# 	msk[0] = 1.
		# 	sent_len = int(np.sum(msk))
		# 	rel_prob = rel_prob[np.arange(len(pred_gold)), pred_gold]
		# 	rel_pred = rel_argmax(rel_prob)
		# 	outputs.append(rel_pred[:sent_len])
		
		for msk, pred_gold, rel_prob in zip(np.transpose(mask), pred_golds.T, rel_probs):
			msk[0] = 1.
			sent_len = int(np.sum(msk))
			rel_prob = rel_prob[np.arange(len(pred_gold)), 0]
			rel_pred = rel_argmax(rel_prob)
			outputs.append(rel_pred[:sent_len])

		return outputs


	def save(self, save_path):
		self.pc.save(save_path)


	def load(self, load_path):
		self.pc.populate(load_path)

