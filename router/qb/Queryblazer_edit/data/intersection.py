import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--train_file', type=str, required=True)
    args = parser.parse_args()
    
    
    # read line by line and find intersection

    with open(args.test_file, 'r') as f:
        file1 = f.readlines()
    with open(args.train_file, 'r') as f:
        file2 = f.readlines()
    

    print(len(file1), len(file2))


    

    file1 = set(file1)
    file2 = set(file2)

    

    print(len(file1), len(file2))
    intersection = file1.intersection(file2)
    # print(intersection)
    print(len(intersection))
    print(len(intersection)/len(file1)*100, "precent of test data is in train data")

if __name__ == '__main__':
    main()