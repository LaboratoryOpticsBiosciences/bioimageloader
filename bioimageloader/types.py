from pathlib import Path
from typing import List

PIL_IMAGE_EXT = [
    '.png',
    '.jpg',
    '.jpeg',
]
TIFFFILE_IMAGE_EXT = [
    '.tif',
    '.tiff',
]
KNOWN_IMAGE_EXT = PIL_IMAGE_EXT + TIFFFILE_IMAGE_EXT
BundledPath = List[Path]
