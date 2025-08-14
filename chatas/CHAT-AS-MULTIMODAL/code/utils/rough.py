from dataset import DialogCCData, Dialog, Utterance, create_image_path_by_url


dialog_data = DialogCCData(
    "data/DialogCC/train.csv",
    to_filter=True,
    to_replace=False,
    image_path_by_url=create_image_path_by_url(
        "../tmp/image_names", "../tmp/images_n"
    ),
    to_unroll=True,
    min_images_per_dialog=1,
    n_samples=1100,
    to_split=False,
)
print(len(dialog_data))