# Training

## Create Training file for MPC
```sh
python code/mpc/create_train.py --inp data/DDC/train.txt --out ./train.mpc
```
## Create Main Trie 

```sh
python code/mpc/main_trie_creation.py --input_file ./train.mpc --output_trie ./main.mpc --threshold 1
```
Note: takes in only sentences with freq>=threshold 

## Create Suffix Trie

```sh
python code/mpc/suffix_trie_creation.py --input_file ./train.mpc --output_trie ./suffix.mpc --suffix_threshold 1
```
Note: takes in only sentences with freq>=suffix_threshold

## Create Test set

```sh
# python data/DDC/create_test.py --n 57 --inp ./data/DDC/test.txt --out ./data/DDC/test_formatted.txt
```
Note: breaks each sentence in test set ```min(n, len(sentence))``` times uniformly randomly through out the sentence while mantaining a ```min_prefix```(default=2) and ```min_suffix```(default=1).

## Predict using MPC

```sh
python code/mpc/mpc_inference.py --input_file data/DDC/seen/test_formatted.txt --main_trie ./main.mpc --suffix_trie ./suffix.mpc --output_file ./completions.mpc
```

## Transform Predictions into Evaluation Format

```sh
python code/mpc/transform2eval.py --input_file ./completions.mpc --output_file ./eval.DDC.mpc
```