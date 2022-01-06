from bioimageloader.base import NucleiDataset

class DSB2018(NucleiDataset):
    acronym = 'DSB2018'

    @property
    def file_list(self):
        ...

    def get_image(self):
        ...
    ...
