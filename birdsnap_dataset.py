import os
import io
from torch.utils.data import Dataset

class BirdsnapDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, coarsef, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            coarsef (string): File containing the coarse labels pertaining to a species
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform

        with open(coarsef, 'r') as fin:
            lines = fin.readlines()

        lines = [line.strip() for line in lines]
        lines = lines[1:]

        fine_index = {}
        coarse_labels = {line.split(',')[2] for line in lines}
        coarse_index = {}

        for coarse_id, coarse_label in enumerate(coarse_labels):
            coarse_index[coarse_label] = coarse_id

        for fine_index, line in enumerate(lines):
            tokens = line.split(',')
            fine_index[tokens[1]] = (fine_index, coarse_index[tokens[2]])

        self.id_index = []

        species = os.listdir(root_dir)

        for specie in species:
            specie_path = os.join(root_dir, specie)
            image_list = os.listdir(specie_path)

            image_path_list = [(os.join(specie_path, image_item), fine_index[specie]) for image_item in image_list]

            self.id_index.extend(image_path_list)

    def __len__(self):
        return len(self.id_index)

    def __getitem__(self, idx):
        
        image = io.imread(self.id_index[idx][0])
        
        sample = {'image': image, 'fine_label': self.id_index[idx][1][0], 'coarse_label': self.id_index[idx][1][1]}

        if self.transform:
            sample = self.transform(sample)

        return sample