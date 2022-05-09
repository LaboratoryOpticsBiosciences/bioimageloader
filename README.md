
# bioimageloader
> _Load bioimages for machine learning applications_

[![Python version](https://img.shields.io/pypi/pyversions/bioimageloader)](https://pypi.org/project/bioimageloader/)
[![PyPI version](https://img.shields.io/pypi/v/bioimageloader)](https://pypi.org/project/bioimageloader/)
[![License](https://img.shields.io/github/license/LaboratoryOpticsBiosciences/bioimageloader)](https://github.com/LaboratoryOpticsBiosciences/bioimageloader/blob/main/LICENSE)

_bioimageloader_ is a python library to make it easy to load bioimage datasets for
machine learning and deep learning. Bioimages come in numerous and inhomogeneous forms.
_bioimageloader_ attempts to wrap them in unified interfaces, so that you can easily
concatenate, perform image augmentation, and batch-load them.

**_bioimageloader_ provides**

1. collections of interfaces for popular and public bioimage datasets
2. image augmentation using [albumentations], which is popular and powerful
   image augmentation library (for 2D images)
3. compatibility with [scikit-learn], [tensorflow], and [pytorch]


## Table of Contents
- [Quick overview](#quick-overview)
    - Load a single dataset
    - Load multiple datasets
    - Batch-load datasets
- [bioimageloader is not/does not](#bioimageloader-is-notdoes-not)
- [Why bioimageloader](#why-bioimageloader)
- [Installation](#installation)
- [Documentation](#documentation)
- [Available collections](#available-collections)
- [QnA](#qna)
- [Contributing](#contributing)
- [Contact](#contact)

## Quick overview
Find full guides at [bioimageloader-docs:User Guides]

1. Load a single dataset

    Load and iterate [_2018 Data Science Bowl_](https://www.kaggle.com/c/data-science-bowl-2018/)

    ```python
    from bioimageloader.collections import DSB2018
    import albumentations as A

    transforms = A.Compose([
        A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])
    dsb2018 = DSB2018('path/to/root_dir', transforms=transforms)
    for data in dsb2018:
        image = data['image']
        mask = data['mask']
    ```

2. Load multiple datasets

    Load DSB2018 and [_Triple Negative Breast Cancer (TNBC)_](https://ieeexplore.ieee.org/document/8438559)

    ```python
    from bioimageloader import Config, ConcatDataset
    from bioimageloader.collections import DSB2018, TNBC
    import albumentations as A

    transforms = A.Compose([
        A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])
    cfg = {
        'DSB2018': { 'root_dir': 'path/to/root_dir' },
        'TNBC'   : { 'root_dir': 'path/to/root_dir' },
    }
    config = Config.from_dict(cfg)
    datasets = config.load_datasets(transforms=transforms)
    cat = ConcatDataset(datasets)
    for meow in cat:
        image = meow['image']
        mask = meow['mask']
    ```

3. Batch-load dataset

    ```python
    from bioimageloader import BatchDataloader

    call_cat = BatchDataloader(cat,
                               batch_size=16,
                               drop_last=True,
                               num_workers=8)
    for meow in call_cat:
        batch_image = meow['image']
        batch_mask = meow['mask']
    ```

## bioimageloader is not/does not

- _not_ a full pipeline for ML/DL
- _not_ a hub to bioimage datasets (if it ever becomes one, it would be awesome though)
- _does not_ host data (only interfaces)
- _does not_ provide one-click links for downloading data
- _does not_ overwrite the source data


## Why _bioimageloader_
_bioimagesloader_ is a by-product of my thesis. This library collected bioimage datasets
for machine learning and deep learning. I needed a lot of diverse bioimages for
self-supervised neural networks for my thesis. While I managed to find many great
datasets, they all came with different folder structures and formats. In addition, I
encountered many issues to load and process them, which were sometimes technical or just
rooted from the nature of bioimages. For instances of technical issues, some datasets
were missing one or two pairs of image and annotation, had broken files, had very
specific file formats that cannot be easily read in python, or provided mask annotation
not in image format but in .xml format. It was rather painful to deal with all these
edge cases one by one. But anyway I did it and I thought it would be valuable to package
and share it with community even though the number of datasets is small for the moment,
so that others do not have to suffer.


## Installation
Install the latest version from PyPI. _bioimageloader_ requires Python 3.8 or higher.
Find more options at [bioimageloader-docs:Installation]

```bash
pip install bioimageloader
```

## Documentation
Full documentation is available at [bioimageloader-docs]


## Available collections
Go to [bioimageloader-docs:Catalogue]


## QnA
### Why no direct download link to each dataset?
_bioimageloader_ provides only codes (interfaces) to load data but not data itself. We
believe that it is important for you to go there, read papers, understand terms and
licenses to **appreciate their works**, because bioimages themselves are sciences and
results of time, efforts, and resources. You still can find links to their project pages
or papers at [bioimageloader-docs:Catalogue], and you need to follow their instruction
to get data. Once you downloaded a dataset and unzipped it, (if it is supported by
_bioimageloader_) you simply pass its root directory as the first argument to
corresponding class from collections `bioimageloader.collections`.

### Dataset that I want is not in the [bioimageloader-docs:Catalogue]
First of all, I named each dataset class rather arbitrary. Try to find the
dataset you want with authors' names or with other keywords (if it has any), and
you may find it having an unexpected name. If it is the case, I apologize for
bad naming.

If you still cannot find it, then you have two options: either you do it yourself (see
below question and please consider contributing!), or you can file an issue so that the
community can help.


### Don't know how to write my own dataloader.
Writing a dataloader requires a bit of Python skills. No easy way. Please read
[templates] carefully and see how others are implemented. File an [issue], and I am
willing to help.


### How to run a ML/DL model?
_bioimageloader_ only helps loading images/annotations, not running ML/DL
models. Still, you may find some useful examples at [bioimageloader-docs:User Guides].
Also check out [ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic).


### I want more granular control over datasets individually
Each bioimage dataset is very unique and it is natural that users want more controls
and it was true for my work as well. Good news is that _bioimageloader_ suggests a
template that you can extend from and make a subclass in your liking. Bad news is
that you need to know how to make a subclass in Python (not so bad I hope. I suppose
that you may have knowledge of Python, if you want to develop ML/DL in Python
anyway). This guide [Modifying existing collections] covers it.


## Contributing
Find guide at [bioimageloader-docs:Contributing]

Also check out [TODO list](./TODO.md).


## Contact
I am open to any feedbacks, suggestions, and discussions. Reach out to me by github or
email.

Seongbin Lim
- Homepage: https://sbinnee.github.io/
- Email: seongbin.lim _at_ polytechnique.edu, sungbin246 _at_ gmail.com

<!-- links -->
[albumentations]: https://albumentations.ai/
[scikit-learn]:  https://scikit-learn.org/stable/index.html
[tensorflow]: https://www.tensorflow.org/
[pytorch]: https://pytorch.org/
[bioimageloader-docs]: https://laboratoryopticsbiosciences.github.io/bioimageloader-docs/
[bioimageloader-docs:Installation]: https://laboratoryopticsbiosciences.github.io/bioimageloader-docs/installation/index.html
[bioimageloader-docs:Catalogue]: https://laboratoryopticsbiosciences.github.io/bioimageloader-docs/catalogue/index.html
[bioimageloader-docs:User Guides]: https://laboratoryopticsbiosciences.github.io/bioimageloader-docs/user_guides/index.html
[templates]:  https://github.com/LaboratoryOpticsBiosciences/bioimageloader/blob/main/bioimageloader/template.py
[issue]: https://github.com/LaboratoryOpticsBiosciences/bioimageloader/issues
[Modifying existing collections]: https://laboratoryopticsbiosciences.github.io/bioimageloader-docs/user_guides/more2_subclassing.html
[bioimageloader-docs:Contributing]: https://laboratoryopticsbiosciences.github.io/bioimageloader-docs/contributing/index.html
