#!bin/bash

mkdir ../../data/MMDD/

wget -P ../../data/MMDD/ https://raw.githubusercontent.com/shh1574/multi-modal-dialogue-dataset/refs/heads/main/dataset/MultiModalDialogue_dev.json
wget -P ../../data/MMDD/ https://raw.githubusercontent.com/shh1574/multi-modal-dialogue-dataset/refs/heads/main/dataset/MultiModalDialogue_test.json
wget -P ../../data/MMDD/ https://raw.githubusercontent.com/shh1574/multi-modal-dialogue-dataset/refs/heads/main/dataset/MultiModalDialogue_train.json

# for num in $(seq -f "%02g" 0 20); 
# do
#     echo "train_$num"
#     wget -P ../../data/MMDD/ https://raw.githubusercontent.com/google-research/google-research/refs/heads/master/multimodalchat/MMDD/train/train_$num.json
# done