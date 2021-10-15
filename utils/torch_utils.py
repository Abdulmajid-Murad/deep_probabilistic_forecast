
import torch
from torch.utils.data import Dataset, DataLoader


class ForecastDataset(Dataset):
    
    def __init__(self, X_train, y_train, sequence_length, sequential, stride=1):
        self.sequence_length = sequence_length
        self.X_train = X_train
        self.y_train = y_train
        self.stride = stride
        self.x_end_idx = self.get_x_end_idx()
        self.sequential = sequential

    def __getitem__(self, index):
        hi = self.x_end_idx[index]
        lo = hi - self.sequence_length
        x = self.X_train[lo:hi]
        y = self.y_train[hi-1]

        x = torch.from_numpy(x).type(torch.float)
        y = torch.from_numpy(y).type(torch.float)
        
        if not self.sequential:
            x = torch.flatten(x)
        
        return x, y

    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        x_index_set = range(self.sequence_length, self.X_train.shape[0])
        x_end_idx = [x_index_set[j * self.stride] for j in range((len(x_index_set)) // self.stride)]
        return x_end_idx


def torch_loader(X_train, y_train, X_test, y_test, sequence_length, batch_size,  sequential=False):

    train_set = ForecastDataset(X_train, y_train, sequence_length, sequential)
    train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0)

    test_set = ForecastDataset(X_test, y_test, sequence_length, sequential)
    test_loader = DataLoader(test_set, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0)

    return train_loader, test_loader


def get_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
        torch.cuda.manual_seed(42)
    else:
        dev="cpu"
    device = torch.device(dev)
    print('Running computation on: ', device)
    return device





