from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from torch.utils.data.dataloader import int_classes, string_classes
from torch._six import container_abcs, string_classes, int_classes
import re
import collections

import torch

np_str_obj_array_pattern = re.compile(r'[SaUO]')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

def collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(collate_err_msg_format.format(elem.dtype))

            return collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    
    elif isinstance(elem, container_abcs.Mapping):
        res = {key: collate([d[key] for d in batch]) for key in elem if key != 'instance_mask'}
        if 'instance_mask' in elem:
            max_obj = max([d['instance_mask'].shape[0] for d in batch])
            instance_mask = torch.zeros(
                len(batch), max_obj, *(elem['instance_mask'].shape[1:]))
            for i in range(len(batch)):
                num_obj = batch[i]['instance_mask'].shape[0]
                instance_mask[i, :num_obj] = torch.from_numpy(
                    batch[i]['instance_mask'])
            res.update({'instance_mask': instance_mask})
        return res
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]

    raise TypeError(collate_err_msg_format.format(elem_type))