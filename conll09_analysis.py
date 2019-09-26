from collections import Counter
import sys

def read_conll(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    conll_data = []
    sents = []
    for line in data:
        if len(line.strip()) == 0:
            if len(sents) > 0:
                conll_data.append(sents)
                sents = []
        else:
            sents.append(line.strip().split('\t'))

    if len(sents) > 0:
        conll_data.append(sents)

    return conll_data


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
            tree.append(Vertex(line[0], line[1], line[5], line[9]))
        for idx in range(len(tree)):
            if tree[idx].head == -1:
                continue
            else:
                tree[tree[idx].head].children.append(idx)
        trees.append(tree)
    return trees


def get_path(tree, pred_id):
    head = tree[pred_id].head
    if head == 0:
        return [0]

    path = []
    while head != 0:
        path.append(tree[head].ID)
        head = tree[head].head

    path.append(0)

    path.reverse()

    return path


def get_position_tag(tree, pos_id):
    head = tree[pos_id].head
    if len(tree[head].children) == 1:
        return 'S'
    else:
        children = tree[head].children
        # children.sort(reverse = False)
        if max(children) == pos_id:
            return 'R'
        elif min(children) == pos_id:
            return 'L'
        else:
            return 'M'


def calculate(tree, pred_id, arg_id):
    pred_path = get_path(tree, pred_id)
    arg_path = get_path(tree, arg_id)

    diff_start = -1
    for idx in range(min(len(pred_path), len(arg_path))):
        if pred_path[idx] != arg_path[idx]:
            break
        else:
            diff_start = idx

    diff_pred_path_len = len(pred_path[diff_start + 1:])
    diff_arg_path_len = len(arg_path[diff_start + 1:])
    # pred_position_tag = get_position_tag(tree, pred_id)
    # arg_position_tag = get_position_tag(tree, arg_id)

    return (diff_pred_path_len, diff_arg_path_len)  # , pred_position_tag, arg_position_tag


def analysis(conll_data):
    trees = build_tree(conll_data)

    position_stat = []

    for idx in range(len(conll_data)):
        sent = conll_data[idx]
        tree = trees[idx]

        # search all predicate
        pred_ids = []
        for jdx in range(len(sent)):
            if sent[jdx][12] == 'Y':
                pred_ids.append(int(sent[jdx][0]))

        if len(pred_ids) == 0:
            continue

        for jdx in range(len(pred_ids)):
            pred_id = pred_ids[jdx]

            for kdx in range(len(sent)):
                if sent[kdx][jdx + 14] != '_':
                    position_stat.append(calculate(tree, pred_id, int(sent[kdx][0])))

    return position_stat


def evaluate(conll_data, rules):
    trees = build_tree(conll_data)

    ca = 0
    pa = 0
    ga = 0

    for idx in range(len(conll_data)):
        sent = conll_data[idx]
        tree = trees[idx]

        # search all predicate
        pred_ids = []
        for jdx in range(len(sent)):
            if sent[jdx][12] == 'Y':
                pred_ids.append(int(sent[jdx][0]))

        if len(pred_ids) == 0:
            continue

        for jdx in range(len(pred_ids)):
            pred_id = pred_ids[jdx]

            for kdx in range(len(sent)):
                if sent[kdx][jdx + 14] != '_':
                    ga += 1
                pred_plen, arg_plen = calculate(tree, pred_id, int(sent[kdx][0]))
                if str(pred_plen) + '-' + str(arg_plen) in rules:  # we detect it as argument
                    pa += 1
                    if sent[kdx][jdx + 14] != '_':
                        ca += 1

    p = ca / pa
    r = ca / ga
    f1 = 2 * p * r / (p + r)
    return p * 100, r * 100, f1 * 100


if __name__ == '__main__':
    training_data = read_conll(sys.argv[1])
    dev_data = read_conll(sys.argv[2])
    test_data = read_conll(sys.argv[3])

    MAX_RULES = int(sys.argv[4]) # 100

    output_file = sys.argv[5]

    position_stats = analysis(training_data)

    c = Counter([str(p[0]) + '-' + str(p[1]) for p in position_stats])

    most_c = c.most_common(MAX_RULES)

    rules = []
    sum = 0
    count = 0
    for item in most_c:
        info = item[0].split('-')
        sum += item[1]
        count += 1
        # print(str(count) + ':', info + [item[1] / len(position_stats) * 100])
        # print(str(count) + '\t' + info[0] + '\t' + info[1])
        rules.append(info[0] + '-' + info[1])

    print('The whole coverage of top %d rules is %f' % (len(rules), sum / len(position_stats) * 100))

    dev_p, dev_r, dev_f1 = evaluate(dev_data, rules)
    test_p, test_r, test_f1 = evaluate(test_data, rules)

    print('dev p:%.2f r:%.2f f1:%.2f' % (dev_p, dev_r, dev_f1))
    print('test p:%.2f r:%.2f f1:%.2f' % (test_p, test_r, test_f1))

    with open(output_file, 'w', encoding='utf-8') as fout:
        for item in rules:
            fout.write(item.replace('-','\t')+'\n')

