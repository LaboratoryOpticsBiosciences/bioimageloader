# bioimageloader
Load bioimages for machine leaning applications
---
`bioimageloader` is a python library to provide templates for bioimage datasets
to develop computer vision deep neural networks. Find supported templates down
below [need tag link].


# Why use `bioimageloader`?
`bioimagesloader` is a by-product of my thesis. This library collected bioimage
datasets for machine learning and deep learning. I needed a lot of diverse
bioimaes for self-supervised neural networks for my thesis. While I managed to
find many great datasets, they all came with different folder structures and
formats. In addition each has its own exceptions rooted from technical issues to
nature of bioimages. For instances of technical issues, some datasets were
missing one or two pairs of image and annotation, had broken files, had very
specific file formats that cannot be easily read in python, or provided mask
annotation not in image format but in .xml format. It was a big pain in the ass
to deal with all these edge cases one by one, but anyway I did it and I thought
it would be valuable to package and share it with community so that others do
not have to suffer.

Wait, I did not mention what sorts of issues are rooted from nature of bioimages
yet. You can find them in transforms section [need tag link].



bioimageloader defines an empty template that predefines necessary attributes
and methods to load datasets for developing deep neural netowrks.


Any deep learning tasks start from loading data.

ImJoy, ZeroCostDL4Mic are model-oriented.

Incoporate NGFF format, meaning loading NGFF-Zarr for DL becomes easy.

Defining a new loader requires knowledge of python class and attributes and
methods.

I hope that bioimageloader can provide a reasonable standard.

Augmentation is done bu imgaug library.


Note
---
1. that the template is a subclass of pytorch dataset. (Can't it be generic?)
   Yes, it can be generic actually! When it becomes a generic loader, it can
   even be easily used with sklearn. (TensorFlow support? IDK, need to check how
   they changed DataLoader)

2. Currently, MaskDataset is a default base since all datasets gathered are
   mainly nuclei/cells datasets.
   - [ ] BBoxDataset
   - [ ] KPointDataset (Not really applicable to bioimages)


## Installation

# dev install (recommended for the moment)
```bash
git clone https://github.com/sbinnee/bioimageloader
cd bioimageloader && pip install -e .
```


## How to use?
`bioimageloader` provides only codes to load data but not data itself. It comes
down to the license issue, since some bioimages may have a complicated procedure
to get, even though they were published. Once you downloaded a dataset and
unzipped it, (if it is supported by `bioimageloader`) you simply pass its root
directory as the first argument to corresponding class from collections
`bioimageloader.collections`, such as below.

```python
from bioimageloader.collections import DSB2018

dset = DSB2018('path/to/root_dir')
for data in dset:
    image = data['image']
    mask = data['mask']
```


## Assumption
- Images have 3 channels

    Images that have grayscale are repeated to have 3 channels, so that they can
    be treated equally with RGB color images during color related transforms.

    Images that have 2 channels are a bit special. In general, they will be
    appended with one additional channel. `bioimageloader` will respect the
    colors of stains applied or the order of channels described if there is any
    (in case of fluorescence microscopy where we can actually see visible
    colors). Otherwise, it will fill RG channels sequentially based on file
    names.

    Image with 3 channels are not treated at all.

    Lastly, images that have more than 3 channels needs some filtering.
    `bioimageloader` provides argument called [`sel_ch` NOT DECIDED YET] to allow
    which channels to select. Optionally, users may want to aggregate all
    channels and make them have grayscale. See `grayscale` and `grayscale_mode`
    in detail.


- Annotation mask has one single channel

    Ensure to have 1 annotation, because usually that is enough. But each
    dataset will provide a way to select one or more channel(s) through
    `image_ch` and `anno_ch` arguments.


- Images have dtype UINT8

    ...



## How to use augmentation with `albumentations`
Albumentations is a popular library for image augmentation and `bioimageloader`
makes use of it through `transforms` argument.


## Dataset that I want is not in the supported list
First of all, I named each dataset class rather arbitrary. Try to find the
dataset you want with authors' names or with other keywords (if it has any), and
you may find it having an unexpected name. If it is the case, I apologize for
bad names.

If you still cannot find it. Then you have two options: either you do it
yourself following the guideline or you can file an issue so that the community
can update it.

1. Use template and implement attributes and methods

2. Test

3. Point path to the dataset

```python
from bioimageloader import MaskDataset

class NewDataset(MaskDataset):
    def get_image(self, ...):
    ...

    def get_mask(self, ...):
    ...

dset = NewDataset('path/to/root_dir')
```


## Available templates
22 datasets

Table
- BBBC002
- BBBC006
- BBBC007
- BBBC008
- BBBC013
- BBBC014
- BBBC015
- BBBC016
- BBBC018
- BBBC020
- BBBC021
- BBBC026
- BBBC039
- BBBC041
- ComputationalPathology
- DSB2018
- DigitalPathology
- FRUNet
- MurphyLab
- S_BSST265
- TNBC
- UCSB


<!-- Put this in another README -->
## Worth mentioning datasets
All datasets are very unique. Read each docstring for details.

- BBBC006
    The same field of view with varying focus. zind=16 is the most in-focus
    plane. Thus it loads only zind=16 by default.


<!-- Put this in another README with more details -->
## I want more granular control over datasets individually
Each bioimage dataset is very unique and it is natural that users want more
controls and it was true for my work as well. Good news is that `bioimageloader`
suggests a template that you can extend from and make a subclass in your liking.
Bad news is that you need to know how to make a subclass in Python (not so bad I
hope. I suppose that you may have knowledge of Python, if you want to develop
ML/DL in Python anyway). I included some examples of subclassing for my use
case. I hope that they are useful with the template.


## QnA
1. Why no download link to each dataset?
    License issue. `bioimageloader` only provides interfaces not data itself.


2. Don't know how to write my own dataloader.
    Writing a dataloader requires a bit of python skills. No easy way. Read
    templates carefully and see how others are implemented. File an issue, and
    I am willing to help.


3. How to run a ML/DL model?
    `bioimageloader` only helps loading images/annotations, not running ML/DL
    models. See ZeroCostDL4Mic.


## Contact
Seongbin Lim
- Homepage: https://sbinnee.github.io/
- Email: seongbin.lim _at_ polytechnique.edu, sungbin246 _at_ gmail.com
