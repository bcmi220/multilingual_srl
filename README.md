# Syntax-aware Multilingual SRL 

## 1. Syntax Pruning Rules Extraction
```
python conll09_analysis.py /path/to/conll09_ENG_train.dataset /path/to/conll09_ENG_dev.dataset /path/to/conll09_ENG_test.dataset 150 ./processed/conll09_ENG_rules.txt
```

## 2. Dataset Preprocessing (format conversion & syntax rules utilization)
```
# baseline: 
python preprocess-conll09.py --train./datasets/conll09-english/conll09_train.dataset --dev ./datasets/conll09-english/conll09_dev.dataset --test ./datasets/conll09-english/conll09_test.dataset --out_dir ./processed/english/

# k-order pruning baseline:
python preprocess-conll09.py --train ./datasets/conll09-english/conll09_ENG_train.dataset --dev./datasets/conll09-english/conll09_ENG_dev.dataset --test ./datasets/conll09-english/conll09_ENG_test.dataset --test_ood ./datasets/conll09-english/conll09_ENG_test_ood.dataset --k_pruning 10 --out_dir./processed/english-10-pruning

# 
python preprocess-conll09.py --train ./datasets/conll09-english/conll09_train.dataset --dev ./datasets/conll09-english/conll09_dev.dataset --test ./datasets/conll09-english/conll09_test.dataset --test_ood ./datasets/conll09-english/conll09_test_ood.dataset --syntax_rules ./processed/english/rules-20.txt --out_dir./processed/english/ --unified

```

## 3. Model training and evaluation
```
python run/train.py --config_file ./configs/config.fasttext.hitelmo.japanese.cfg --dynet-gpu
```

