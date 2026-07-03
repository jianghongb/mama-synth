import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    if getattr(opt, 'dataset_mode', 'aligned') == 'mha':
        from data.mha_dataset import MhaDataset
        dataset = MhaDataset()
    elif getattr(opt, 'dataset_mode', 'aligned') == 'mha_perimage_norm':
        from data.mha_perimage_norm_dataset import MhaPerImageNormDataset
        dataset = MhaPerImageNormDataset()
    elif getattr(opt, 'dataset_mode', 'aligned') == 'mha_baseline':
        from data.mha_baseline_dataset import MhaBaselineDataset
        dataset = MhaBaselineDataset()
    else:
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
