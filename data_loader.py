import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler, DistributedSampler


class MemmapDataset(Dataset):
    def __init__(self, data_dir, dataset_type, block_size, dtype=np.uint16):

        data = np.memmap(os.path.join(data_dir, f'{dataset_type}.bin'), mode='c', dtype=dtype)

        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.block_size = block_size
        self.dtype = dtype
        self.lenght = len(data)-block_size

        data.flush()
        del data

    def __len__(self):
        return self.lenght

    def __getitem__(self, idx):
        data = np.memmap(os.path.join(self.data_dir, f'{self.dataset_type}.bin'), mode='c', dtype=self.dtype)
        d = data[idx:idx+self.block_size+1]
        x = torch.from_numpy(d[:-1]).type(torch.int32)
        y = torch.from_numpy(d[1:]).type(torch.int64)
        data.flush()
        d.flush()
        del data
        del d
        del idx
        return x,y

class RandomBatchSampler(BatchSampler):
    def __init__(self, data_source, batch_size, batches_to_load):
        super().__init__(data_source, batch_size, drop_last=False)
        self.data_source = data_source
        self.batch_size = batch_size
        self.batches_to_load = batches_to_load

    def __iter__(self):
        # Random batch indices creation at each epoch
        for i in range(self.batches_to_load):
            batch_idx = torch.randint(len(self.data_source), (self.batch_size,))
            yield batch_idx.tolist()
            del batch_idx

    def __len__(self):
        # Number of batches per epoch
        return self.batches_to_load

class RandomDistributedBatchSampler(DistributedSampler):
    def __init__(self, data_source, batch_size, batches_to_load):
        super().__init__(data_source, shuffle=True, seed=0, drop_last=False) # It is not needed to specify num_replicas because it is retrieved from the current distributed group
        self.data_source = data_source
        self.batch_size = batch_size
        self.batches_to_load = batches_to_load

    def __iter__(self):
        # Random batch indices creation at each epoch and for each replica
        for _ in range(self.batches_to_load):
            # Randomly sample batch_size indices from the rank's subset
            batch_idx = torch.randint(len(self.data_source), (self.batch_size//self.num_replicas,))
            yield batch_idx.tolist()

    def __len__(self):
        # Number of batches per epoch
        return self.batches_to_load


def custom_data_loader(data_dir,dataset_type,block_size,batch_size,batches_to_load,num_workers,pin_memory=True):
    dataset    = MemmapDataset(data_dir, dataset_type=dataset_type, block_size=block_size)
    sampler    = RandomBatchSampler(dataset, batch_size=batch_size,batches_to_load=batches_to_load)
    dataloader = DataLoader(dataset,
                            batch_sampler=sampler,
                            num_workers=num_workers,
                            pin_memory=pin_memory,)
    return dataloader

def custom_distributed_data_loader(data_dir,dataset_type,block_size,batch_size,batches_to_load,num_workers,pin_memory=True):
    dataset    = MemmapDataset(data_dir, dataset_type=dataset_type, block_size=block_size)
    sampler    = RandomDistributedBatchSampler(dataset, batch_size=batch_size,batches_to_load=batches_to_load)
    dataloader = DataLoader(dataset,
                            batch_sampler=sampler,
                            num_workers=num_workers,
                            pin_memory=pin_memory,)
    return dataloader