from prox import GroupProximalOperator, torch
import warnings
import multiprocessing
from functools import partial

warnings.warn('Prox_cl is abandand,please use prox_cl_acc instead')

__all__ = ['ProxL2_1', 'ProxL2_1over2', 'ProxL2_2over3', 'ProxMCP', 'ProxSCAD', 'ProxCappedl1', 'convert_prox_tex',
           'ProxL1_1over2']

convert_prox_tex = {
    'ProxL2_1': r'$\ell_{2,1}$',
    'ProxL2_1over2': r'$\ell_{2,\frac{1}{2}}$',
    'ProxL2_2over3': r'$\ell_{2,\frac{2}{3}}$',
    'ProxMCP': r'group MCP',
    'ProxSCAD': r'group SCAD',
    'ProxCappedl1': r'group Capped $\ell_1$',
    'ProxL1_1over2': r'$\ell_{1,\frac{1}{2}}$'
}


class ProxL2_1(GroupProximalOperator):
    def __init__(self, n: int, gLen: int):
        super().__init__()
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)

    @classmethod
    def obj(cls, x: torch.Tensor, xtilde: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('Not implemented yet')

    def prox(self, x: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x_subvectors = self._split_vectors(x)
        norms = torch.stack([subvec.norm(dim=1, p=2) for subvec in x_subvectors], dim=0)
        tao = torch.where(norms <= v, 0.0, torch.nn.functional.relu(1 - v / norms))
        x_subvectors = [torch.mul(tao[i].reshape(-1, 1), x_subvectors[i]) for i in range(self.num_subvectors)]
        return self._concentrate_vectors(x_subvectors)

    @classmethod
    def name(cls) -> str:
        return 'ProxL2_1'

    def _split_vectors(self, x: torch.Tensor) -> list[torch.Tensor]:
        return torch.chunk(x, self.num_subvectors, dim=1)

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x_subvectors, dim=1)


class ProxL2_1over2(GroupProximalOperator):
    def __init__(self, n: int, gLen: int):
        super().__init__()
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)

    @classmethod
    def obj(cls, x: torch.Tensor, xtilde: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('Not implemented yet')

    # def prox(self, x: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    #     x_subvectors = self._split_vectors(x)
    #     norms = torch.stack([subvec.norm(dim=1, p=2) for subvec in x_subvectors], dim=0)
    #     break_point = 1.5 * v ** (2 / 3)
    #     tao = torch.where(
    #         norms.detach() <= break_point.detach(),
    #         0.0, (2 / 3)*(1.0 + torch.cos((2 / 3) * (
    #             torch.arccos(
    #                 torch.clip(-3 ** 1.5 / 4 * v * (norms ** -1.5), min=torch.tensor(  -1 + 0.00001).cuda(),   max=torch.tensor( 1 - 0.000001).cuda())
    #             ))))
    #     )
    #
    #     x_subvectors = [torch.mul(tao[i].reshape(-1, 1), x_subvectors[i]) for i in range(self.num_subvectors)]
    #     x = torch.cat(x_subvectors, dim=1)
    #     return x
    def prox(self, x: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x_subvectors = self._split_vectors(x)
        norms = torch.stack([subvec.norm(dim=1, p=2) for subvec in x_subvectors], dim=0)
        break_point = 1.5 * v.detach() ** (2 / 3)

        tao = torch.where(
            torch.logical_or(norms.detach() <= break_point.detach(),
                             -3 ** 1.5 / 4 * v.detach() * (norms.detach() ** -1.5) <= -1),
            0.0,
            (2 / 3) * (1.0 + torch.cos((2 / 3) * (torch.arccos(
                torch.clip(-3 ** 1.5 / 4 * v * (norms ** -1.5),
                           min=torch.tensor(-1.0).cuda(),
                           max=torch.tensor(0.0).cuda()
                           ))
            )))
        )
        x_subvectors = [torch.mul(tao[i].reshape(-1, 1), x_subvectors[i]) for i in range(self.num_subvectors)]
        return self._concentrate_vectors(x_subvectors)

    @classmethod
    def name(cls) -> str:
        return 'ProxL2_1over2'

    def _split_vectors(self, x: torch.Tensor) -> list[torch.Tensor]:
        return torch.chunk(x, self.num_subvectors, dim=1)

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x_subvectors, dim=1)


class ProxL2_2over3(GroupProximalOperator):
    def __init__(self, n: int, gLen: int):
        super().__init__()
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)

    @classmethod
    def obj(cls, x: torch.Tensor, xtilde: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('Not implemented yet')

    def prox(self, x: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x_subvectors = self._split_vectors(x)
        norms = torch.stack([subvec.norm(dim=1, p=2) for subvec in x_subvectors], dim=0)
        norms = torch.where(
            (1 / 256 - (8 * (v.detach() ** 3) / (729 * norms.detach() ** 4))) <= 0,
            torch.sqrt(torch.sqrt(8 * 256 * (v.detach() ** 3) / 729)) + 1e-10 * v.detach(),
            norms)
        tmp = torch.sqrt(1 / 256 - (8 * (v ** 3) / (729 * norms ** 4)))
        t = (1 / 16 + tmp) ** (1 / 3) + (1 / 16 - tmp) ** (1 / 3)
        tao = torch.where(norms.detach() <= 2 * ((2 / 3) * v.detach() ** 0.75),
                          0.0,
                          (1.0 / 8.0) * (
                                  torch.sqrt(2.0 * t) + (torch.sqrt(2.0 / (torch.sqrt(2.0 * t)) - 2.0 * t))) ** 3)
        x_subvectors = [torch.mul(tao[i].reshape(-1, 1), x_subvectors[i]) for i in range(self.num_subvectors)]
        return self._concentrate_vectors(x_subvectors)

    @classmethod
    def name(cls) -> str:
        return 'ProxL2_2over3'

    def _split_vectors(self, x: torch.Tensor) -> list[torch.Tensor]:
        return torch.chunk(x, self.num_subvectors, dim=1)

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x_subvectors, dim=1)


class ProxMCP(GroupProximalOperator):
    def __init__(self, n: int, gLen: int, gamma: float):
        super().__init__()
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)
        self.gamma = gamma
        assert self.gamma > 1, ValueError('gamma must be greater than 1')

    def prox(self, x: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x_subvectors = self._split_vectors(x)
        norms = torch.stack([subvec.norm(dim=1, p=2) for subvec in x_subvectors], dim=0)
        tao = torch.where(
            norms.detach() <= self.gamma * v.detach(),
            torch.nn.functional.relu(1 - v / norms) * self.gamma / (self.gamma - 1),
            1
        )
        x_subvectors = [torch.mul((tao[i]).reshape(-1, 1), x_subvectors[i]) for i in
                        range(self.num_subvectors)]
        return self._concentrate_vectors(x_subvectors)

    @classmethod
    def name(cls):
        return 'ProxMCP'

    def _split_vectors(self, x: torch.Tensor) -> list[torch.Tensor]:
        return torch.chunk(x, self.num_subvectors, dim=1)

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x_subvectors, dim=1)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class ProxSCAD(GroupProximalOperator):
    def __init__(self, n: int, gLen: int, gamma: float):
        super().__init__()
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)
        self.gamma = gamma
        assert self.gamma > 2, ValueError('gamma must be greater than 2')

    def prox(self, x: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x_subvectors = self._split_vectors(x)
        norms = torch.stack([subvec.norm(dim=1, p=2) for subvec in x_subvectors], dim=0)
        gamma = self.gamma
        # tao = torch.where((norms <= self.gamma * v),
        #                   torch.nn.functional.relu(1 - v / norms) * self.gamma /(self.gamma - 1), 1)torch.where(norms <= v, 0.0, (1 - v / norms))
        tao = torch.where(
            norms.detach() <= 2 * v,
            torch.nn.functional.relu(1 - v / norms),
            torch.where(norms <= gamma * v,
                        torch.nn.functional.relu(1 - (gamma * v / ((gamma - 1) * norms))) * (gamma - 1) / (gamma - 2),
                        1.0)
        )
        x_subvectors = [torch.mul(tao[i].reshape(-1, 1), x_subvectors[i]) for i in range(self.num_subvectors)]
        return self._concentrate_vectors(x_subvectors)

    @classmethod
    def name(cls):
        return 'ProxSCAD'

    def _split_vectors(self, x: torch.Tensor) -> list[torch.Tensor]:
        return torch.chunk(x, self.num_subvectors, dim=1)

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x_subvectors, dim=1)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class ProxCappedl1(GroupProximalOperator):
    def __init__(self, n: int, gLen: int, gamma: float):
        super().__init__()
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)
        self.gamma = gamma

    #  assert self.gamma > 0.5, ValueError('gamma must be greater than ')

    def prox(self, x: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x_subvectors = self._split_vectors(x)
        norms = torch.stack([subvec.norm(dim=1, p=2) for subvec in x_subvectors], dim=0)
        gamma = self.gamma
        tao = torch.where((norms <= gamma + v / (2 * gamma)),
                          torch.nn.functional.relu(1 - v / (self.gamma * norms)), 1)
        x_subvectors = [torch.mul(tao[i].reshape(-1, 1), x_subvectors[i]) for i in range(self.num_subvectors)]
        return self._concentrate_vectors(x_subvectors)

    @classmethod
    def name(cls):
        return 'ProxCappedl1'

    def _split_vectors(self, x: torch.Tensor) -> list[torch.Tensor]:
        return torch.chunk(x, self.num_subvectors, dim=1)

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x_subvectors, dim=1)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


import numpy as np
import math


class ProxL1_1over2(GroupProximalOperator):
    def __init__(self, n: int, gLen: int):
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)

    # self.num_samples: int = num_samples

    def prox(self, x: torch.Tensor, v, *args, **kwargs) -> torch.Tensor:
        #

        batch_size, total_features = x.size()
        #
        features_per_subvector = total_features // self.num_subvectors
        # x_subvectors = torch.chunk(x, batch_size, dim=0)
        x_subvectors = torch.chunk(x.reshape(-1), batch_size * self.num_subvectors, dim=-1)

        con1 = 3 * v.detach() ** (2 / 3) / (2 ** (4 / 3))
        con2 = 1.5 * v.detach() ** (2 / 3)

        s_arange = torch.arange(1, features_per_subvector + 1, device=x.device)
        v_reshape = v.reshape(-1)
        con4 = (3 * v_reshape ** (2 / 3) * (0.5 ** (4 / 3))) * s_arange ** (2 / 3)

        # for persample in x_subvectors:
        #     chk = torch.chunk(persample, self.num_subvectors, dim=1)
        #     mask.extend([self.L1_1over2_part.prox(n.reshape(-1), v, con1, con2) for n in chk])
        par = partial(self.L1_1over2_part.prox, nu=v_reshape, con1=con1, con2=con2, con4=con4, s_arange=s_arange)
        # [par(group) for group in x_subvectors]
        return torch.concatenate(list(map(par, x_subvectors))).reshape(batch_size,
                                                                       total_features)  # self._concentrate_vectors(x_subvectors)

    def _split_vectors(self, x: torch.Tensor) -> list[torch.Tensor]:
        return torch.chunk(x, self.num_subvectors, dim=1)

    @classmethod
    def name(cls):
        return 'ProxL1_1over2'

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError('You dont need to call this method')

    class L1_1over2_part:

        # @staticmethod
        # def obj(y: torch.Tensor, ytilde: torch.Tensor, nu) -> torch.Tensor:
        #     return (nu * torch.pow(ytilde.sum(), 1 / 2) + 0.5 * (
        #             torch.linalg.norm(y - ytilde, 2) ** 2)).detach().cpu().numpy()
        @staticmethod
        def obj(y: torch.Tensor, ytilde: torch.Tensor, nu: float) -> torch.Tensor:
            # 直接返回 PyTorch 张量，避免不必要的 CPU 转换
            return nu * torch.sqrt(ytilde.sum()) + 0.5 * torch.norm(y - ytilde) ** 2

        @classmethod
        def prox(cls, x: torch.Tensor, nu: torch.Tensor, con1, con2, con4, s_arange) -> torch.Tensor:
            # 记录顺序和绝对值，用于后续计算
            x_abs = torch.abs(x)
            sign_x = torch.sign(x)
            ys_l1 = x_abs.sum()

            if ys_l1 <= con1:  # case1
                return x * 0
            else:
                indices = torch.argsort(-x_abs)
                y = x_abs[indices]
                if y[-1] > con2:  # case2
                    return (x_abs - cls._calculate_cs(ys_l1, nu, len(x.detach()))) * sign_x
                else:
                    sorted_indices = torch.argsort(indices)
                    # Search s
                    s_list = [x * 0]
                    s_list.extend(cls._loop_search_s(y, nu, con4, s_arange))
                    Js_list = [cls.obj(y, per_ytilde, nu) for per_ytilde in s_list]
                    # cls._rearrange_org(s_list[-1], sorted_indices, sign_x).reshape(-1)
                    return cls._rearrange_org(s_list[torch.argmin(torch.tensor(Js_list))], sorted_indices, sign_x)
                # return cls._rearrange_org(s_list[1:][np.argmin(Js_list[1:])], sorted_indices, sign_x)

        @staticmethod
        def _rearrange_org(ytilde: torch.Tensor, sorted_indices, sign: torch.Tensor) -> torch.Tensor:
            return ytilde[sorted_indices] * sign

        @classmethod
        def _loop_search_s(cls, y: torch.Tensor, nu: torch.Tensor, con4, s_arange) -> list[torch.Tensor]:
            tmp = torch.real(cls._calculate_condition3(v=nu, q=1 / 2, s=s_arange).reshape(-1))
            #  res = (y_l1_array >= condition4) & (y_l1_array >= tmp) & (y > cs_array) & (
            #          cs_array >= torch.tensor(y_plus, dtype=y.dtype, device=y.device))
            # Search s
            y_l1_array = y.cumsum(dim=0)
            cs_array = cls._calculate_cs(y_l1_array, nu, s_arange)
            new_yl1 = torch.cat((y[1:8], torch.tensor([-1], device=y.device)))
            # #torch.tensor(np.append(y.detach().cpu().numpy(), -1)[1:], dtype=y.dtype, device=y.device))
            res = (y_l1_array >= con4) & (y_l1_array >= tmp) & (y > cs_array) & (
                     cs_array >= new_yl1)
            return [torch.relu(y - cs_array[ind]) for ind in torch.nonzero(res.flatten(), as_tuple=True)[0]]  # list(np.where(res.detach().cpu().numpy().reshape(-1))[0])]

        @staticmethod
        def _calculate_cs(y_l1_array: torch.Tensor, nu: torch.Tensor, s) -> torch.Tensor:
            return (math.sqrt(3.0) * nu) / (
                    4 * torch.sqrt(y_l1_array) * torch.cos(
                (1 / 3) * torch.arccos(
                    torch.clip(-3 * math.sqrt(3.0) * nu * s * torch.pow(y_l1_array, -3 / 2) / 4, min=-1, max=1))))

        @staticmethod
        def _calculate_condition3(v, q, s):
            return (2 - q) * (v * q * s / ((q - 1) ** (1 - q))) ** (1 / (2 - q))
