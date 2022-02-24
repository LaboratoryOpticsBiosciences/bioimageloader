from pathlib import Path
from typing import List, TypeVar, NewType, Any

T = TypeVar('T')

# Keep all ext in lower case
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
Bundled = List[T]
BundledPath = Bundled[Path]
