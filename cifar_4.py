from torch.utils.data import Dataset
import os
import pickle
import numpy as np

class CIFARDataset(Dataset):
    def __init__(self, root='data', train=True):
        data_path = os.path.join(root, 'cifar-10-batches-py')
        if train:
            data_files = [os.path.join(data_path, "data_batch_{}".format(i)) for i in range(1, 6)]
        else:
            data_files = [os.path.join(data_path, "test_batch")]
        self.images=[]
        self.labels=[]
        for data_file in data_files:
            with open(data_file, 'rb') as fo:
                data = pickle.load(fo, encoding='bytes') #giải mã file
                self.images.extend(data[b'data'])
                self.labels.extend(data[b'labels'])

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, item):
        image = self.images[item].reshape((3, 32, 32)).astype(np.float32)
        label = self.labels[item]
        return image/255., label

if __name__ == '__main__':
    dataset = CIFARDataset(root='cifar-10-batches-py', train=True )
    image, label = dataset.__getitem__(123)
    print(image.shape)