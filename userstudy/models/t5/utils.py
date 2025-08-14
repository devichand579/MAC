import torch

CRED = "\033[91m"
CEND = "\033[0m"


def suffix_encoder(tokenizer, text, max_length, batching=False, prev_space=True):
    if batching:
        for i in range(len(text)):
            if not prev_space[i]:
                text[i] = "«" + text[i]
    else:
        if not prev_space:
            text = "«" + text
    encoded = tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    if batching:
        for i in range(len(encoded["input_ids"])):
            if not prev_space[i]:
                encoded["input_ids"][i] = torch.cat(
                    (
                        encoded["input_ids"][i][1:],
                        torch.tensor([tokenizer.pad_token_id]),
                    )
                )
                encoded["attention_mask"][i] = torch.cat(
                    (encoded["attention_mask"][i][1:], torch.tensor([0]))
                )
        return encoded
    if not prev_space:
        encoded["input_ids"][0] = torch.cat(
            (encoded["input_ids"][0][1:], torch.tensor([tokenizer.pad_token_id]))
        )
        encoded["attention_mask"][0] = torch.cat(
            (encoded["attention_mask"][0][1:], torch.tensor([0]))
        )
    return encoded


def suffix_decoder(tokenizer, encoded):
    text = tokenizer.decode(
        torch.cat((torch.tensor([673]), encoded), dim=0), skip_special_tokens=True
    )
    text = text[1:]
    # if(len(text)==0):
    #     print("Empty text")
    #     print(encoded)
    return text


def prefix_encoder(tokenizer, text, max_length, batch=False):
    if batch:
        for i in range(len(text)):
            if text[i][-1] == " ":
                text[i] = text[i][:-1] + "<tspace>"
    else:
        if text[-1] == " ":
            text = text[:-1] + "<tspace>"
    encoded = tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return encoded


def merge_prefix_suffix(prefix, suffix):
    if len(suffix) > 0 and len(prefix) > 0 and suffix[0] == " " and prefix[-1] == " ":
        return prefix[:-1] + suffix
    else:
        return prefix + suffix
