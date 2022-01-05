# bioimageloader
bioimageloader is a python library to provide templates for bioimage datasets to
develop computer vision deep neural networks. Find supported templates down
below.

bioimageloader defines a template to implement necessary attributes and
methods to run deep neural netowrks.

Note that the template is a subclass of pytorch dataset. (Can't it be generic?)
Yes, it can be generic actually! When it becomes a generic loader, it can even
be easily used with sklearn.
(TensorFlow support? IDK, need to check)

Any deep learning tasks start from loading data.

ImJoy, ZeroCostDL4Mic are model-oriented.

Incoporate NGFF format, meaning loading NGFF-Zarr for DL becomes easy.



## Installation

# dev install (recommended)
```bash
git clone
cd gitrepo && pip install -e .
```


## Usage
1. Download a dataset (and unzip it, if necessary)
2. Point path to the dataset
```python
from bioimageloader import DSB2018

dset = DSB2018('path/to/root_dir')
data = dset[0]
image = data['image']
mask = data['mask']

```


## Available templates

Table
- DSB2018
- ...


## QnA
1. Why no download link to each dataset?

    License issue. bioimageloader only provides interfaces not data itself.
    Already supported data should be free to use for research.

2. Don't know how to write my own dataloader.

    Writing a dataloader requires a bit of python skills. No easy way. Read
    templates carefully and see how others are implemented. File an issue, and
    I am willing to help.

3. How to run a ML/DL model?

    bioimageloader only helps loading images not running ML/DL models.
