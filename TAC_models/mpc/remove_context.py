import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", help="Path to the context train file")
    parser.add_argument("--final_file", help="Path to the file containing only utterances")
    args = parser.parse_args()

    with open(args.train_file, 'r') as f:
        with open(args.final_file, 'w') as final_f:  # Open the final file for writing
            for line in f:
                last_eou_index = line.rfind('\t')  # Find the index of last <eou>
                context = line[:last_eou_index]  # Context is everything before the last <eou>
                utterance = line[last_eou_index:].strip()  # Extract the utterance
                utterance = utterance.strip()
                
                # Write the utterance to the final file
                final_f.write(utterance + '\n')
