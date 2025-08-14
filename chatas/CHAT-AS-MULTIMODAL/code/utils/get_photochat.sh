#!bin/bash

mkdir ../../data/PhotoChat/

wget -P ../../data/PhotoChat/ https://raw.githubusercontent.com/google-research/google-research/refs/heads/master/multimodalchat/photochat/dev/dev_00.json
wget -P ../../data/PhotoChat/ https://raw.githubusercontent.com/google-research/google-research/refs/heads/master/multimodalchat/photochat/dev/dev_01.json
wget -P ../../data/PhotoChat/ https://raw.githubusercontent.com/google-research/google-research/refs/heads/master/multimodalchat/photochat/test/test_00.json
wget -P ../../data/PhotoChat/ https://raw.githubusercontent.com/google-research/google-research/refs/heads/master/multimodalchat/photochat/test/test_01.json

for num in $(seq -f "%02g" 0 20); 
do
    echo "train_$num"
    wget -P ../../data/PhotoChat/ https://raw.githubusercontent.com/google-research/google-research/refs/heads/master/multimodalchat/photochat/train/train_$num.json
done