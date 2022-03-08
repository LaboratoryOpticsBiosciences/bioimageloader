# Todo for the first Alpha Release v0.1.0
I2K 2022 *“developing open source image analysis platforms/tools”*
- Abstract til 1st March
- Event 6-10 May (virtual)
- https://forum.image.sc/t/i2k-2022-conference-talk-workshop-submissions-are-open/62833

### Zarr
- [ ]  Zarr and in-house data [experimental]

### Docs
- [x]  Overview table
    1. `.md` for maintaining
    2. `.html` (use table gen service, which I don’t like it... but whatever)
    3. put sample image links
    4. embed in docs and github readme
- [x]  Docs, notebook examples
- [x]  Module docs
- [x]  Quickstart
- [x]  clean README.md
- [ ]  indicate datasets that need pre-parse somehow?

### Utils
- [x]  Data vis
- [x]  Models
    - [x]  bioimage.io
- [x]  Run and eval models
    - [ ]  Summary table which model excels in which dataset
- [x]  CommonDataset, CommonMaskDataset
- [x]  re-ordering channels
        np.ascontiguousarray, 'C' order, arr[..., (2, 1, 0)]
- [x]  random sampling, shuffle in BatchDataLoader
- [ ]  Metrics for benchmarking (StarDist has done a great job, their license is BSD-3)
- [ ]  (maybe nope) Download scripts

### Fix
- [x]  Take out those that do not have mask anno and put them in `Dataset`
    - [x]  Implement `__getitem__` for `Dataset`
    - [x]  Change base for them
- [x]  Fix data[’mask’]  # (b, h, w) → (b, 1, h, w)? (necessary?)
    Not really necessary? Just have them (b, h, w) for now
- [x]  Fix data['mask'].dtype == bool
    When mask has a single channel, make them have the same dtype.
    Albumentations supports only UINT8 and FLOAT32.
    - bool -> uint8
    - or int16 (because why not... mask.dtype does not matter)
- [x]  fix plt.rcParams['image.interpolation'] does not work in `./notebooks/_sample_images.ipynb`
    cv2.resize was an issue, not matplotlib
- [x]  update overview table
    - [x]  Reordered
    - [x]  Add missing ones
        - [x]  BBBC041
- [x]  consistent `root_dir`. Now it is a mess. We want something described in the
  `docs/user_guides/basic0_prepare_datasets.rst`.
- [ ]  number of images, including test sets
- [ ]  substitute `num_calls` with `num_samples`? (I think samples sound right)
- [ ]  Load all anno types, if there are more than one (e.g. BBBC007)

### Others
- [ ]  Migrate repo to LOB account and open to public
- [ ]  More data
    - StarDist (subset of DSB2018)
    - Cellpose
    - https://twitter.com/cmci_/status/1498156970700046337?t=9n2gDTpeM1Oq3JcCOV9QOg
        All 3 are related. Rooted from one and modified somehow?
        1. Romain Guiet. (2022). HeLa "Kyoto" cells under the scope (Version v0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6139958
        2. Romain Guiet, & Olivier Burri. (2022). Cellpose models for Label Prediction from Brightfield and Digital Phase Contrast images (Version v0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6140111
        3. Romain Guiet. (2022). Automatic labelling of HeLa "Kyoto" cells using Deep Learning tools (Version v0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6140064
    - CRCHisto
    - CEM500K
    - [ ]  OpenCell [https://opencell.czbiohub.org/](https://opencell.czbiohub.org/)

## For later
- [ ]  AnnoDatasetInterface
    Interface for Dataset with annotation, base for all Dataset with annotation.
    Currently attr like `anno_dict` is not an abstract property, though it should be
    once we have other types, BboxDataset for instance,
- [ ]  BboxDataset
    - [ ]  BBBC041
    - [ ]  and more
- [ ]  3D [experimental]
    - [ ]  need 3D augmentation lib
- [ ]  Custom augmentation ex) 5 channels
- [ ]  time-series


<!-- Put this in another README -->
## Worth mentioning datasets
All datasets are very unique. Read each docstring for details.

- BBBC006
    The same field of view with varying focus. zind=16 is the most in-focus
    plane. Thus it loads only zind=16 by default.
