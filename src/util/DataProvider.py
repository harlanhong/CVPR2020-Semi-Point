from torch.utils.data.dataloader import _DataLoaderIter as DataLoaderIter
import torch
class DataProvider:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataiter = DataLoaderIter(self.dataloader)
        self.iteration = 0  # 当前epoch的batch数
        self.epoch = 0  # 统计训练了多少个epoch
    def next(self):
        try:
            batch = self.dataiter.next()
            self.iteration += 1
            return batch

        except StopIteration:  # 一个epoch结束后reload
            self.epoch += 1
            self.dataiter = DataLoaderIter(self.dataloader)
            self.iteration = 1  # reset and return the 1st batch

            batch = self.dataiter.next()
            return batch