from abc import ABCMeta
import torch

__all__ = ['ProximalOperator', 'GroupProximalOperator', 'torch']



class ProximalOperator(metaclass=ABCMeta):
    @classmethod
    def obj(cls, x: torch.Tensor, xtilde: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def prox(cls, x: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def name(cls):
        return cls.__name__

    def __call__(self, *args, **kwargs):
        return self.prox(*args, **kwargs)


class GroupProximalOperator(ProximalOperator):
    n: int
    gLen: int
    num_subvectors: int
    num_samples: int

    __slots__ = ('n', 'gLen', 'num_subvectors', 'num_samples')

    @staticmethod
    def get_num_subvectors(n: int, gLen: int) -> int:
        return n // gLen

    @classmethod
    def obj(cls, x: torch.Tensor, xtilde: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def prox(self, x: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def name(cls):
        raise NotImplementedError

    def _split_vectors(self, x: torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
