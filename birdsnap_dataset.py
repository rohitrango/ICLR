import os
import io

class BirdSnapDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform

        self.id_index = []

        species = os.listdir(root_dir)

        for index, specie in enumerate(species):
            specie_path = os.join(root_dir, specie)
            image_list = os.listdir(specie_path)

            image_path_list = [(os.join(specie_path, image_item), index) for image_item in image_list]

            self.id_index.extend(image_path_list)

    def __len__(self):
        return len(self.id_index)

    def __getitem__(self, idx):
        
        image = io.imread(self.id_index[idx][0])
        
        sample = {'image': image, 'label': self.id_index[idx][1]}

        if self.transform:
            sample = self.transform(sample)

        return sample