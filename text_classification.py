import torch
from datasets import load_dataset


class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.dataset = load_dataset(path='seamew/ChnSentiCorp', split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']

        return text, label


if __name__ == '__main__':
    dataset = Dataset('train')

    len(dataset), dataset[:10]