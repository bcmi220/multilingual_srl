CUDA_VISIBLE_DEVICES=1 python extract_features_hdf5.py --input_file=../conllx/conllx-en-sentences.txt --output_file=../conllx/conllx-en-sentences.txt.bert.large.cased.hdf5 --vocab_file=../bert-models/cased_L-24_H-1024_A-16/vocab.txt --bert_config_file=../bert-models/cased_L-24_H-1024_A-16/bert_config.json --init_checkpoint=../bert-models/cased_L-24_H-1024_A-16/bert_model.ckpt --layers=-2 --max_seq_length=512 --batch_size=8

CUDA_VISIBLE_DEVICES=1 python extract_features_hdf5.py --input_file=../conllx/conllx-ar-sentences.txt --output_file=../conllx/conllx-ar-sentences.txt.bert.multi.cased.hdf5 --vocab_file=../bert-models/multi_cased_L-12_H-768_A-12/vocab.txt --bert_config_file=../bert-models/multi_cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=../bert-models/multi_cased_L-12_H-768_A-12/bert_model.ckpt --layers=-2 --max_seq_length=512 --batch_size=8

CUDA_VISIBLE_DEVICES=1 python extract_features_hdf5.py --input_file=../conllx/conllx-bg-sentences.txt --output_file=../conllx/conllx-bg-sentences.txt.bert.multi.cased.hdf5 --vocab_file=../bert-models/multi_cased_L-12_H-768_A-12/vocab.txt --bert_config_file=../bert-models/multi_cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=../bert-models/multi_cased_L-12_H-768_A-12/bert_model.ckpt --layers=-2 --max_seq_length=512 --batch_size=8

CUDA_VISIBLE_DEVICES=1 python extract_features_hdf5.py --input_file=../conllx/conllx-cs-sentences.txt --output_file=../conllx/conllx-cs-sentences.txt.bert.multi.cased.hdf5 --vocab_file=../bert-models/multi_cased_L-12_H-768_A-12/vocab.txt --bert_config_file=../bert-models/multi_cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=../bert-models/multi_cased_L-12_H-768_A-12/bert_model.ckpt --layers=-2 --max_seq_length=512 --batch_size=8

CUDA_VISIBLE_DEVICES=1 python extract_features_hdf5.py --input_file=../conllx/conllx-da-sentences.txt --output_file=../conllx/conllx-da-sentences.txt.bert.multi.cased.hdf5 --vocab_file=../bert-models/multi_cased_L-12_H-768_A-12/vocab.txt --bert_config_file=../bert-models/multi_cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=../bert-models/multi_cased_L-12_H-768_A-12/bert_model.ckpt --layers=-2 --max_seq_length=512 --batch_size=8

CUDA_VISIBLE_DEVICES=1 python extract_features_hdf5.py --input_file=../conllx/conllx-de-sentences.txt --output_file=../conllx/conllx-de-sentences.txt.bert.multi.cased.hdf5 --vocab_file=../bert-models/multi_cased_L-12_H-768_A-12/vocab.txt --bert_config_file=../bert-models/multi_cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=../bert-models/multi_cased_L-12_H-768_A-12/bert_model.ckpt --layers=-2 --max_seq_length=512 --batch_size=8

CUDA_VISIBLE_DEVICES=1 python extract_features_hdf5.py --input_file=../conllx/conllx-es-sentences.txt --output_file=../conllx/conllx-es-sentences.txt.bert.multi.cased.hdf5 --vocab_file=../bert-models/multi_cased_L-12_H-768_A-12/vocab.txt --bert_config_file=../bert-models/multi_cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=../bert-models/multi_cased_L-12_H-768_A-12/bert_model.ckpt --layers=-2 --max_seq_length=512 --batch_size=8

CUDA_VISIBLE_DEVICES=1 python extract_features_hdf5.py --input_file=../conllx/conllx-ja-sentences.txt --output_file=../conllx/conllx-ja-sentences.txt.bert.multi.cased.hdf5 --vocab_file=../bert-models/multi_cased_L-12_H-768_A-12/vocab.txt --bert_config_file=../bert-models/multi_cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=../bert-models/multi_cased_L-12_H-768_A-12/bert_model.ckpt --layers=-2 --max_seq_length=512 --batch_size=8

CUDA_VISIBLE_DEVICES=1 python extract_features_hdf5.py --input_file=../conllx/conllx-nl-sentences.txt --output_file=../conllx/conllx-nl-sentences.txt.bert.multi.cased.hdf5 --vocab_file=../bert-models/multi_cased_L-12_H-768_A-12/vocab.txt --bert_config_file=../bert-models/multi_cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=../bert-models/multi_cased_L-12_H-768_A-12/bert_model.ckpt --layers=-2 --max_seq_length=512 --batch_size=8

CUDA_VISIBLE_DEVICES=1 python extract_features_hdf5.py --input_file=../conllx/conllx-pt-sentences.txt --output_file=../conllx/conllx-pt-sentences.txt.bert.multi.cased.hdf5 --vocab_file=../bert-models/multi_cased_L-12_H-768_A-12/vocab.txt --bert_config_file=../bert-models/multi_cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=../bert-models/multi_cased_L-12_H-768_A-12/bert_model.ckpt --layers=-2 --max_seq_length=512 --batch_size=8

CUDA_VISIBLE_DEVICES=1 python extract_features_hdf5.py --input_file=../conllx/conllx-sl-sentences.txt --output_file=../conllx/conllx-sl-sentences.txt.bert.multi.cased.hdf5 --vocab_file=../bert-models/multi_cased_L-12_H-768_A-12/vocab.txt --bert_config_file=../bert-models/multi_cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=../bert-models/multi_cased_L-12_H-768_A-12/bert_model.ckpt --layers=-2 --max_seq_length=512 --batch_size=8

CUDA_VISIBLE_DEVICES=1 python extract_features_hdf5.py --input_file=../conllx/conllx-sv-sentences.txt --output_file=../conllx/conllx-sv-sentences.txt.bert.multi.cased.hdf5 --vocab_file=../bert-models/multi_cased_L-12_H-768_A-12/vocab.txt --bert_config_file=../bert-models/multi_cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=../bert-models/multi_cased_L-12_H-768_A-12/bert_model.ckpt --layers=-2 --max_seq_length=512 --batch_size=8

CUDA_VISIBLE_DEVICES=1 python extract_features_hdf5.py --input_file=../conllx/conllx-tr-sentences.txt --output_file=../conllx/conllx-tr-sentences.txt.bert.multi.cased.hdf5 --vocab_file=../bert-models/multi_cased_L-12_H-768_A-12/vocab.txt --bert_config_file=../bert-models/multi_cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=../bert-models/multi_cased_L-12_H-768_A-12/bert_model.ckpt --layers=-2 --max_seq_length=512 --batch_size=8

CUDA_VISIBLE_DEVICES=1 python extract_features_hdf5.py --input_file=../conllx/conllx-zh-sentences.txt --output_file=../conllx/conllx-zh-sentences.txt.bert.chinese.cased.hdf5 --vocab_file=../bert-models/chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=../bert-models/chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=../bert-models/chinese_L-12_H-768_A-12/bert_model.ckpt --layers=-2 --max_seq_length=512 --batch_size=8