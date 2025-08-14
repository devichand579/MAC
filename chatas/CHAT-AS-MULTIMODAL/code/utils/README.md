### Dataset


### How to Use

1. **Create Image Path by URL Mapping**  
   Before loading the dataset, you may want to map image URLs to local image paths. This can be done using the `create_image_path_by_url()` function.
   ```python
   image_path_by_url = create_image_path_by_url(
       image_names_dir="../tmp/image_names", 
       images_dir="../tmp/images"
   )
   ```

2. **Load and Process the Dataset**  
   Initialize a `DialogCCData` object by providing the dataset file path and optional configurations. For example, the `decided test data` can be obtained using

   ```python
   dialog_data = DialogCCData(
       path="data/DialogCC/test.csv",
       to_filter=True,
       to_replace=True,
       image_path_by_url=image_path_by_url,
       to_unroll=True,
       min_images_per_dialog=1,
       n_samples=1100,
       to_split=True
   )
   print(f"Total dialogs: {len(dialog_data)}")
   ```
   Each element in `dialog_data` is a Tuple containing a `Dialog` object (context + prefix) and suffix.

**NOTE**
Verify that the hash of the `decided test data` is `5b2f271ea09dbac9c6a94ce94a74516706d81fc23c21a36f0323b22594ae29a4`


### Features Summary
Each of the following are applied in the order listed below:

- **Filtering**: Remove dialogs whose images don’t exist locally.
- **Replacing URLs**: replace image URLs with local file paths.
- **Unrolling**: Turn each dialog into several partial dialogs.
- **minimum images per dialog**: Filter out dialogs with fewer images than the specified threshold.
- **Sampling**: Select a fixed number of dialogs from the dataset (from each category).
- **Splitting**: Split the last utterance in a dialog at various positions to create new dialogs and corresponding suffixes.



### Outline of Classes

#### 1. `Utterance`
Represents a single utterance within a dialog.
- **Attributes:**
  - `text`: The textual content of the utterance.
  - `images`: A list of image URLs associated with the utterance.
  - `speaker`: The speaker who made the utterance (optional).

#### 2. `Dialog`
Represents a conversation consisting of multiple utterances.
- **Attributes:**
  - `idx`: An identifier for the dialog.
  - `utterances`: A list of `Utterance` objects representing the conversation.
- **Important Methods:**
  - `unroll()`: Unrolls the dialog into a list of partial dialogs, gradually revealing more utterances.
  - `create_splits()`: Splits the last utterance into various versions and creates corresponding dialogs.
  - `context`: Returns all but the last utterance in the dialog.
  - `response`: Returns the last utterance in the dialog.
  - `character_count`: Returns the total number of characters across all utterances' text.
  
#### 3. `DialogCC`
Inherits from `Dialog` and is specialized to handle DialogCC dialogs

#### 4. `DialogData`
Represents a dataset of dialogs.

- **Attributes:**
  - `dialogs`: A list of `Dialog` objects.
  - `suffixes`: A list of suffixes corresponding to the dialogs. (is `None` when `to_split` is `False`)