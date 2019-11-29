# Data Handling

This module handles all the processing of the data for the model. The preprocessing is very important as without it, the model will not learn.

Rather than writing a new preprocessor for each database that we wanted to use, an abstract preprocessor class found in `Preprocessor.py` was created. A preprocessor for each database can then implement this class. Examples of this for RAVDESS, SAVEE, and TESS are found in `RAVDESS_preprocessor.py`, `SAVEE_preprocessor.py`, and `TESS_preprocessor.py` respectively.

## Preprocessor.py Documentaion

Any implementation of the abstract preprocessor class will have the following methods
* `get_audio_data()`: loads the raw audio from a directory of `.wav` files into a pandas dataframe
* `pitch(data_array)`: randomly changes the pitch to a given raw audio data array
* `white_noise(data_array)`: randomly adds white noise to a given raw audio data array
* `shift(data_array)`: preforms a random shift in time to a given raw audio data array
* `volume(data_array)`: randomly increases the volume to a given raw audio data array
* `augment(df, frac=0.1)`: preforms each of the four different types of audio augmentation described above to a random fraction of the total data equal to the argument `frac`
* `mfcc_conversion(df)`: Preforms the MFCC conversion to each of the data in the dataframe
* `split_data(df, le=None, append=True, equalize=True)`: This is the most complex method in the class, preforming the following transformations on the data
  * Integer encodes each of the labels (since we are using Cross Entropy Loss, PyTorch one-hot encodes the labels for us). If the arguemnt `le` is None, then it saves the mapping of the labels to use for later data. Otherwise, it uses the mapping provided by `le`
  * Removes the all samples recorded by actors specified by the class implementation to be used in the testing dataset
  * Splits the remaining data into each of its classes and samples those datasets separately to ensure the train/validation split has equal class distribution
  * Randomly samples the original dataset to obtain an overfit dataset
  * If `equalize` is set to True, then if does another check to make sure the distribution between classes are equal,and if it's not then it drops data from other classes to force them to be equal
  * Prints the distribution of each dataset so we can verify the class distribution is equal
  * Saves each dataset to two tsv files: `data/<dataset>_data.tsv` and `data/<dataset>_label.tsv`. If `append` is set to False then it will overwrite the tsv file. If `append` is set to True then it will add the data to the end of the tsv file.
  * Saves the metadata of the dataset in a json file `data/Metadata.json`
  
## Implementing a Preprocessor Class

For examples implementations, see `RAVDESS_preprocessor.py`, `SAVEE_preprocessor.py`, and `TESS_preprocessor.py`. I will outline exactly what you need to specify if you would like to add an additional dataset.

An overview of the problems we need to solve goes as follows. First, you will have found a dataset that you would like to add and will have downloaded it. Once downloaded and unzipped you will have a folder with some name containing data in some arbitrary format. We need to convert this arbitrary format into a standard format that is easy for the preprocessor to use. Now we have a list of files whose names correspond to metadata about the audio file. This is also in some arbitrary format. We need to tell the proprocessor how to parse the file name and extract the relevant information.

### 1. Create `<DATASET>_metadata.json`
It's always good practice to have information about the database you are using. The major role of this metadata file is to specify the mapping of emotions. For example, some databases label the emotion "fear" as "fearful", or some databases will label each emotion with a number. Instead of cluttering up a function with a bunch of if-statements, it's much cleaner and easier to specify a dictionary and just always use that as a conversion. Name this json file `raw_data/<DATASET>_metadata.json` and you must at least include the following in it:
```json
{
    "emotion": {
        "...": "neutral",
        "...": "happy",
        "...": "sad",
        "...": "angry",
        "...": "fear",
        "...": "disgust",
        "...": "surprised"
    }
}
```
where each `"..."` is how the dataset labels those emotions.

### 2. Create a file called `<DATASET>_preprocessor.py` of the following form 
```python
from .Preprocessor import Preprocessor

import os
import shutil
import json

class DATASET_Preprocessor(Preprocessor):

    name = "<DATASET>_Preprocessor"
    dataset = "<DATASET>"

    def __init__(self, raw_data_dir="...", data_dir="<DATASET>", metadata_file="<DATASET>_metadata.json", seed=None, n_mfcc=30):
        pass

    def rearrange(self):
        pass

    def parse_file_name(self, f_name):
        pass
```
We just need to implement these three functions in the correct way and then we get all the functionallity of the `Preprocessor` class for free.

### 2. Implementing the `__init__()`

Essentially the goal of the init is to specify all the paths needed to find the data and exclude anything you do not want to process.

```python
def __init__(self, raw_data_dir="...", data_dir="<DATASET>", metadata_file="<DATASET>_metadata.json", seed=None, n_mfcc=30):
    Preprocessor.__init__(self, seed=seed, n_mfcc=n_mfcc)     # inherit from Preprocessor class, just copy-paste this in

    self.extra += ['...']
    self.test_actors = ['...', '...']
    
    # Note: you do not need to change these, just copy-paste
    self.original_path = os.path.join(self.ROOT, "raw_data", raw_data_dir)
    self.path = os.path.join(self.ROOT, "raw_data", data_dir)

    metadata_path = os.path.join(self.ROOT, "raw_data", metadata_file)
    self.Metadata = json.load(open(metadata_path, "r"))
```

The only things you need to change are the following
* replace the word `<DATASET>` with whatever the name of your database is
* `raw_data_dir="..."`: This is the name of the directory the data is in from when you initially downloaded the data
* `self.extra += ['...']`: This is a list of any files or folders in the dataset folder from above that you do not want the preprocessor to process
* `self.test_actors = ['...']`: This is a list of actors (using the labels used by the database) that you want to reserve for the testing set (i.e. not samples recorded by them will appear in the training or validation set)

### 3. Implementing `rearrange()`

The data folder that was downloaded is in an arbitrary order and we need to put it in a more convient arrangement. In particular, if there are nested folders we want to flatten them so we just have a single directory containing a long list of files. Note: often information about files are stored in the nested folders that contain the files, so you may need to rename the file to add this information that would otherwise be lost.

```python
def rearrange(self):    
    dir_list1 = self.create_new_data_directory()

    try:
        for var1 in dir_list1: # loops through the actor
            dir_list2 = os.listdir(os.path.join(self.original_path, var1))
            for var2 in dir_list2:
                dir_list3 = os.listdir(os.path.join(self.original_path, var1, var2))
                for var3 in dir_list3:
                    #... continue for as many nested folders as you need to go
                    fname_list = os.listdir(os.path.join(self.original_path, var1, var2, var3, ...))
                    fname_list.sort()
                    for f in fname_list:
                        shutil.copy(os.path.join(self.original_path, actor, f), self.path)
                        # If you need to rename the file
                        new_f = "..."
                        os.rename( os.path.join(self.path, f), os.path.join(self.path, new_f) )
    except Exception as e:
        # data files already in the correct configuration
        pass

    self.check_dataset_created()
```

* `create_new_data_directory()` creates the directory stored in `self.path` specified by the `__init__()`
* Then we loop through the nested folders and copy each file to the directory stored in `self.path` and renaming the file is necessary
* Finally, `check_dataset_created()` prints the number of files in the directory stored in `self.path`. If this number is equal to the total size of the database then you know the rearrange was successful
* The purpose of the try-except statement is so that calling `rearrange()` if the rearrange has already happened does not throw an error. It just makes the code cleaner later.

### 4. Implementing `parse_file_name()`

Now we have a flattened list of files whose file names give all relevant metadata about the file. We need to tell the preprocessor how to parse this filename to extract that information. 

```python
def parse_file_name(self, f_name):
        #           0       1      2
        # parts = ["...", "...", "..."]
        parts = f_name.split('.')[0].split('-')
        gender = self.Metadata["gender"][part[...]]
        emotion = self.Metadata["emotion"][parts[...]]
        actor = part[...]

        skip = ...
        
        return skip, actor, gender, emotion
```

This can be a little confusing so look at the other implemented preprocessors for reference. Essentially you name your file in such a way that is easy to parse, for example separating information by a '-' or '_'. Then knowing the order of that information you can extract it from the file name. `skip` allows you to tell the preprocessor to not process a file.

## Using the Implemented Preprocessor
Now once all those functions are correctly written, you can use the preprocessor as follows:

```python
DATASET = DATASET_preprocessor()
DATASET.rearrange()
df = DATASET.get_audio_data()
df = DATASET.augment(df, frac=0.1)
df = DATASET.mfcc_conversion(df)
le = DATASET.split_data(df, le=None, equalize=True, append=False)
```

See `preprocess.py` to see how to string multiple dataset together.
