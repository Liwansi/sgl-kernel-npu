"""Split storage with combined snapshots for the existing numerical oracles.

Only oracle reads concatenate the planes. The operator and captured graphs
receive persistent, independently allocated K/V tensors and DRAM pointers.
"""

import torch


class SplitKVFixture:
    def __init__(self, combined):
        self.k = combined[..., :512].contiguous()
        self.v = combined[..., 512:].contiguous()
        self.k_ptr = self.k.data_ptr()
        self.v_ptr = self.v.data_ptr()

    @classmethod
    def host(cls, combined, ops, device_id):
        result = cls(combined)
        for name, width in (("k", 512), ("v", 64)):
            tensor, _, pointer = ops.create_shm_tensor(
                (*combined.shape[:-1], width),
                combined.dtype,
                device_id=device_id,
                name=f"split_fixture_{id(result)}_{name}",
            )
            tensor.copy_(getattr(result, name))
            setattr(result, name, tensor)
            setattr(result, f"{name}_ptr", pointer)
        return result

    def combined(self):
        return torch.cat((self.k, self.v), dim=-1)

    def cpu(self):
        return torch.cat((self.k.cpu(), self.v.cpu()), dim=-1)

    def clone(self):
        result = object.__new__(type(self))
        result.k, result.v = self.k.clone(), self.v.clone()
        result.k_ptr, result.v_ptr = result.k.data_ptr(), result.v.data_ptr()
        return result

    def copy_(self, other):
        if isinstance(other, SplitKVFixture):
            self.k.copy_(other.k)
            self.v.copy_(other.v)
        else:
            self.k.copy_(other[..., :512])
            self.v.copy_(other[..., 512:])
        return self

    def fill_row(self, index, value):
        self.k[index].fill_(value)
        self.v[index].fill_(value)

    def __getitem__(self, index):
        return self.combined()[index]

    def __getattr__(self, name):
        return getattr(self.combined(), name)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return func(
            *(x.combined() if isinstance(x, cls) else x for x in args), **(kwargs or {})
        )
