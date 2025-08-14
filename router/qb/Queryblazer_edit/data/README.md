n is decided as the average length of sentences in test set.

To get stats of a file

```sh
python data/get_stats.py --inp data/DDC/test.txt
```

For creating DSTC7 test formatted
```sh
python data/create_test.py --inp data/DSTC7/test.txt --out_dir data/DSTC7 --train_file data/DSTC7/train.txt
```

For creating DDC test formatted
```sh
python data/create_test.py --inp data/DDC/test.txt --out_dir data/DDC --train_file data/DDC/train.txt
```