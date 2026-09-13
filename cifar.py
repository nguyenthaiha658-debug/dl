from torch.utils.data import Dataset
import os
import pickle

class CIFARDataset(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        if train:
            data_files = [os.path.join(self.root, "data_batch_{}".format(i)) for i in range(1, 6)]
        else:
            data_files = [os.path.join(self.root, "test_batch")]
        self.images=[]
        self.labels=[]
        for data_file in data_files:
            with open(data_file, 'rb') as fo:
                data = pickle.load(fo, encoding='bytes')
                self.images.extend(data[b'data'])
                self.labels.extend(data[b'labels'])
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, item):
        image = self.images[item]
        label = self.labels[item]
        return image, label

if __name__ == '__main__':
    dataset = CIFARDataset(root='cifar-10-batches-py', train=True )
    image, label = dataset.__getitem__(123)
    print(image.shape)