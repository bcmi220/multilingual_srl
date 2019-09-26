
USE_SYN_INDEX = 9 # predict syntax # 8 for golden syntax

def read_conll(filename):
	data = []
	sentence = []
	with open(filename, 'r') as fp:
		for line in fp:
			if len(line.strip()) > 0:
				sentence.append(line.strip().split())
			else:
				data.append(sentence)
				sentence = []
		if len(sentence) > 0:
			data.append(sentence)
			sentence = []
	return data

class Vertex:
    def __init__(self, ID, word, pos, head):
        self.ID = int(ID)
        self.word = word
        self.pos = pos
        self.head = int(head)
        self.children = []

def build_tree(conll_data):
    trees = []
    for sent in conll_data:
        tree = [Vertex('0', '<ROOT>', '<ROOT>', '-1')]
        for line in sent:
            tree.append(Vertex(line[0], line[1], line[5], line[USE_SYN_INDEX])) # use predict syntax to build tree
        for idx in range(len(tree)):
            if tree[idx].head == -1:
                continue
            else:
                tree[tree[idx].head].children.append(idx)
        trees.append(tree)
    return trees

def get_syn_path(tree, pos_id):
    head = tree[pos_id].head
    if head == 0:
        return [0]

    path = []
    while head != 0:
        path.append(tree[head].ID)
        head = tree[head].head

    path.append(0)

    path.reverse()

    return path

def calculate_family_path(tree, pred_id, arg_id):
    pred_path = get_syn_path(tree, pred_id)
    arg_path = get_syn_path(tree, arg_id)

    diff_start = -1
    for idx in range(min(len(pred_path), len(arg_path))):
        if pred_path[idx] != arg_path[idx]:
            break
        else:
            diff_start = idx

    diff_pred_path_len = len(pred_path[diff_start + 1:])
    diff_arg_path_len = len(arg_path[diff_start + 1:])

    return (diff_pred_path_len, diff_arg_path_len)

def get_k_son_list(sentence, K):
	# record the syntactic son for every node.(include dummy ROOT)
	syntactic_son_list = [[[] for _ in range(len(sentence)+1)] for _ in range(K)]
	for oidx in range(K):
		for i in range(len(sentence)):
			if oidx == 0:
				syntactic_son_list[oidx][int(sentence[i][USE_SYN_INDEX])].append(int(sentence[i][0])) # use predict syntax to build tree
			else:
				for k in range(len(syntactic_son_list[oidx-1])):
					if int(sentence[i][USE_SYN_INDEX]) in syntactic_son_list[oidx-1][k]:
						syntactic_son_list[oidx][k].append(int(sentence[i][0]))
						break
	return syntactic_son_list

def get_available_list_by_k_son(sentence, k_son_list, current_idx):

	# for this predicate we do pruning by syntactic grammar.
	current_node_idx = int(sentence[current_idx][0])

	reserve_set = set()

	while True:
		for item in k_son_list:
			reserve_set.update(item[current_node_idx])

		if current_node_idx != 0:
			current_node_idx = int(sentence[current_node_idx-1][USE_SYN_INDEX]) # use predict syntax to build tree
		else:
			break 
	
	return reserve_set

def srl2ptb(origin_data, syntax_rule, k_pruning, force_mask, unified):
	trees = build_tree(origin_data)
	srl_data = []
	# pdisamb = 0
	# gdisamb = 0
	# allp = 0

	disamb_set = set()

	diff_set = set()
	for idx in range(len(origin_data)):
		sentence = origin_data[idx]
		tree = trees[idx]
		arg_idx = 0

		if k_pruning != 0:
			k_son_list = get_k_son_list(sentence, k_pruning)
		
		for i in range(len(sentence)):
			if sentence[i][12] == 'Y':
				# we add syntax mask here, for syntax rule we masked the argument to 0
				srl_sent = []

				# allp += 1

				# if sentence[i][2] == sentence[i][13]:
				# 	gdisamb += 1
				# 	diff_set.add(sentence[i][13])

				# if sentence[i][3] == sentence[i][13]:
				# 	pdisamb += 1
				# 	diff_set.add(sentence[i][13])

				if unified:
					disamb_set.add(sentence[i][13].split('.')[1])
					if syntax_rule is not None:
						srl_sent.append(['0','<DUMMY>', '<DUMMY>', '<DUMMY>', '<DUMMY>', '_', 
										sentence[i][0], sentence[i][13].split('.')[1],' _', '_', '1']) # the dummy node must be argument of predicate to get the relation
					else:
						if force_mask:
							srl_sent.append(['0','<DUMMY>', '<DUMMY>', '<DUMMY>', '<DUMMY>', '_', 
										sentence[i][0], sentence[i][13].split('.')[1],' _', '_', '1'])
						else:
							srl_sent.append(['0','<DUMMY>', '<DUMMY>', '<DUMMY>', '<DUMMY>', '_', 
										sentence[i][0], sentence[i][13].split('.')[1],' _', '_']) # the dummy node must be argument of predicate to get the relation


				if k_pruning != 0:
					k_pruning_available_list = get_available_list_by_k_son(sentence, k_son_list, i)
				
				for j in range(len(sentence)):
					token = sentence[j]
					if syntax_rule is not None or k_pruning != 0:

						mask = 0

						if syntax_rule is not None:
							pl, al = calculate_family_path(tree, i+1, j+1) # because ID = index + 1 due to <ROOT> node in tree
							if str(pl)+'-'+str(al) in syntax_rule:
								mask = mask or 1

						if k_pruning != 0:
							if j+1 in k_pruning_available_list:
								mask = mask or 1

						srl_sent.append([token[0], token[1], token[3], token[4], token[5], 
										'_', sentence[i][0] if unified else str(int(sentence[i][0])-1), token[14 + arg_idx], '_', '_', str(mask)])
					else:
						if force_mask:
							srl_sent.append([token[0], token[1], token[3], token[4], token[5], 
											'_', sentence[i][0] if unified else str(int(sentence[i][0])-1), token[14 + arg_idx], '_', '_', '1'])
						else:
							srl_sent.append([token[0], token[1], token[3], token[4], token[5], 
											'_', sentence[i][0] if unified else str(int(sentence[i][0])-1), token[14 + arg_idx], '_', '_'])
				srl_data.append(srl_sent)
				arg_idx += 1

	# print(allp, gdisamb, pdisamb)
	# with open('./test.log','a') as f:
	# 	f.write('\n'.join(list(diff_set))+'\n\n')
	if unified:
		print('disamb set size:', len(list(disamb_set)))
	return srl_data

def save(srl_data, path):
	with open(path,'w') as f:
		for sent in srl_data:
			for token in sent:
				f.write('\t'.join(token))
				f.write('\n')
			f.write('\n')

def make_sentences(train_data, dev_data, test_data, test_ood_data, output_dir):
	sentences = set()
	if train_data is not None:
		for sentence in train_data:
			sentences.add(' '.join([item[1].lower() for item in sentence]))
	if dev_data is not None:
		for sentence in dev_data:
			sentences.add(' '.join([item[1].lower() for item in sentence]))
	if test_data is not None:
		for sentence in test_data:
			sentences.add(' '.join([item[1].lower() for item in sentence]))
	if test_ood_data is not None:
		for sentence in test_ood_data:
			sentences.add(' '.join([item[1].lower() for item in sentence]))
	sentences = list(sentences)
	with open(output_dir, 'w') as f:
		for line in sentences:
			f.write(line+'\n')

def load_syntax_rules(rule_path):
	with open(rule_path, 'r') as f:
		data = f.readlines()
	data = [line.strip().split('\t') for line in data if len(line.strip())>0]
	syn_rules = set()
	for line in data:
		syn_rules.add(line[0]+'-'+line[1])
	return syn_rules

import argparse, os
if __name__ == '__main__':
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--train', default=None)
	argparser.add_argument('--test', default=None)
	argparser.add_argument('--dev', default=None)
	argparser.add_argument('--test_ood', default=None)
	argparser.add_argument('--syntax_rules', default=None)
	argparser.add_argument('--k_pruning', type=int, default=0)
	argparser.add_argument('--out_dir', default='processed')
	argparser.add_argument('--unified', action='store_true')
	args, extra_args = argparser.parse_known_args()

	if not os.path.exists(args.out_dir):
		os.mkdir(args.out_dir)

	syntax_rules = None
	if args.syntax_rules:
		syntax_rules = load_syntax_rules(args.syntax_rules)

	train_conll = None
	if args.train:
		train_conll = read_conll(args.train)
		train_srl = srl2ptb(train_conll, syntax_rules, args.k_pruning, True, args.unified)
		save(train_srl, '%s/train_pro' % args.out_dir)
	
	dev_conll = None
	if args.dev:
		dev_conll = read_conll(args.dev)
		dev_srl = srl2ptb(dev_conll, syntax_rules, args.k_pruning, True, args.unified)
		save(dev_srl, '%s/dev_pro' % args.out_dir)
		os.system('cp %s %s/dev_raw' % (args.dev, args.out_dir))
	
	test_conll = None
	if args.test:
		test_conll = read_conll(args.test)
		test_srl = srl2ptb(test_conll, syntax_rules, args.k_pruning, True, args.unified)
		save(test_srl, '%s/test_pro' % args.out_dir)
		os.system('cp %s %s/test_raw' % (args.test, args.out_dir))
	
	test_ood_conll = None
	if args.test_ood:
		test_ood_conll = read_conll(args.test_ood)
		test_ood_srl = srl2ptb(test_ood_conll, syntax_rules, args.k_pruning, True, args.unified)
		save(test_ood_srl, '%s/test_ood_pro' % args.out_dir)
		os.system('cp %s %s/test_ood_raw' % (args.test_ood, args.out_dir))

	make_sentences(train_conll, dev_conll, test_conll, test_ood_conll, '%s/sentences.txt' % args.out_dir)
	
