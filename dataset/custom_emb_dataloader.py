import torch
from torch.utils.data import DataLoader
    
class CustomEmbDataLoaderSingle(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, pin_memory=True, drop_last=False):
        collate_fn = self.defined_collate
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
            pin_memory=pin_memory
        )

    def defined_collate(self, batch):
        sequences, label_indices = zip(*batch)

        input_seq = torch.stack(sequences)
        labels_task = torch.tensor(label_indices)

        return (input_seq, labels_task)