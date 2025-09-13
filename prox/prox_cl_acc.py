from prox import GroupProximalOperator, torch
from functools import partial
import math

__all__ = ['ProxL2_1', 'ProxL2_1over2', 'ProxL2_2over3', 'ProxMCP', 'ProxSCAD', 'ProxCappedl1', 'convert_prox_tex',
           'ProxLogSum', 'ProxTransformedL1', 'ProxCappedL1over2', 'ProxCappedTransformedL1', 'ProxArctan','ProxL1_1over2','ProxL1_2over3']

# convert_prox_tex = {
#     'ProxL2_1': r'$\ell_{2,1}$',
#     'ProxL2_1over2': r'$\ell_{2,\frac{1}{2}}$',
#     'ProxL2_2over3': r'$\ell_{2,\frac{2}{3}}$',
#     'ProxMCP': r'group MCP',
#     'ProxSCAD': r'group SCAD',
#     'ProxCappedl1': r'group Capped $\ell_1$',
#     'ProxLogSum': r'group Logsum',
#     'ProxTransformedL1':r'T$\ell_1$',
#     'ProxCappedL1over2':r'group Capped $\ell_\frac{1}{2}$'
# }
convert_prox_tex = {
    'ProxL2_1': r'$\ell_{2,1}$',
    'ProxL2_1over2': r'$\ell_{2,\frac{1}{2}}$',
    'ProxL2_2over3': r'$\ell_{2,\frac{2}{3}}$',
    'ProxMCP': r'$L_{2,{\rm{MCP}}}$',
    'ProxSCAD': r'$L_{2,{\rm{SCAD}}}$',
    'ProxCappedl1': r'$L_{2,{\rm{CapL1}}}$',
    'ProxLogSum': r'$L_{2,{\rm{LOG}}}$',
    'ProxTransformedL1': r'$L_{2,{\rm{T}}\ell_1}$',
    'ProxCappedL1over2': r'$L_{2,{\rm{CapL\frac{1}{2}}}}$',
    'ProxL1_1over2': r'$\ell_{1,\frac{1}{2}}$',
    'ProxL1_2over3':r'$\ell_{1,\frac{2}{3}}$',
    'ProxCappedTransformedL1': r'captl1',
    'ProxArctan': r'$\ell_{2,Arctan}$'
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
        batch_size, total_features = x.size()
        features_per_subvector = total_features // self.num_subvectors
        reshaped_x = x.reshape(batch_size, self.num_subvectors, features_per_subvector)
        norms = torch.norm(reshaped_x, p=2, dim=2)  # 形状为 (batch_size, num_subvectors)
        tao = torch.where(norms <= v, 0.0, torch.nn.functional.relu(1 - v / norms))  # 形状为 (batch_size, num_subvectors)
        x_subvectors = tao.unsqueeze(2) * reshaped_x  # 形状为 (batch_size, num_subvectors, features_per_subvector)
        return x_subvectors.reshape(batch_size, total_features)  # 直接重塑为原始形状

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

    def prox(self, x: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        batch_size, total_features = x.size()
        features_per_subvector = total_features // self.num_subvectors
        reshaped_x = x.reshape(batch_size, self.num_subvectors, features_per_subvector)
        norms = torch.norm(reshaped_x, p=2, dim=2)  # 形状为 (batch_size, num_subvectors)

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
        x_subvectors = tao.unsqueeze(2) * reshaped_x  # 形状为 (batch_size, num_subvectors, features_per_subvector)
        return x_subvectors.reshape(batch_size, total_features)  # 直接重塑为原始形状

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
        batch_size, total_features = x.size()
        features_per_subvector = total_features // self.num_subvectors
        reshaped_x = x.reshape(batch_size, self.num_subvectors, features_per_subvector)
        norms = torch.norm(reshaped_x, p=2, dim=2)  # 形状为 (batch_size, num_subvectors)
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
        x_subvectors = tao.unsqueeze(2) * reshaped_x  # 形状为 (batch_size, num_subvectors, features_per_subvector)
        return x_subvectors.reshape(batch_size, total_features)  # 直接重塑为原始形状

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
        batch_size, total_features = x.size()
        features_per_subvector = total_features // self.num_subvectors
        reshaped_x = x.reshape(batch_size, self.num_subvectors, features_per_subvector)
        norms = torch.norm(reshaped_x, p=2, dim=2)  # 形状为 (batch_size, num_subvectors)
        tao = torch.where(
            norms.detach() <= self.gamma * v.detach(),
            torch.nn.functional.relu(1 - v / norms) * self.gamma / (self.gamma - 1),
            1
        )
        x_subvectors = tao.unsqueeze(2) * reshaped_x  # 形状为 (batch_size, num_subvectors, features_per_subvector)
        return x_subvectors.reshape(batch_size, total_features)  # 直接重塑为原始形状

    @classmethod
    def name(cls):
        return 'ProxMCP'

    def _split_vectors(self, x: torch.Tensor) -> list[torch.Tensor]:
        return torch.chunk(x, self.num_subvectors, dim=1)

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x_subvectors, dim=1)


class ProxSCAD(GroupProximalOperator):
    def __init__(self, n: int, gLen: int, gamma: float):
        super().__init__()
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)
        self.gamma = gamma
        assert self.gamma > 2, ValueError('gamma must be greater than 2')

    def prox(self, x: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        batch_size, total_features = x.size()
        features_per_subvector = total_features // self.num_subvectors
        reshaped_x = x.reshape(batch_size, self.num_subvectors, features_per_subvector)
        norms = torch.norm(reshaped_x, p=2, dim=2)  # 形状为 (batch_size, num_subvectors)
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
        x_subvectors = tao.unsqueeze(2) * reshaped_x  # 形状为 (batch_size, num_subvectors, features_per_subvector)
        return x_subvectors.reshape(batch_size, total_features)  # 直接重塑为原始形状

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
        batch_size, total_features = x.size()
        features_per_subvector = total_features // self.num_subvectors
        reshaped_x = x.reshape(batch_size, self.num_subvectors, features_per_subvector)
        norms = torch.norm(reshaped_x, p=2, dim=2)  # 形状为 (batch_size, num_subvectors)
        gamma = self.gamma
        tao = torch.where((norms <= gamma + v / (2 * gamma)), torch.nn.functional.relu(1 - v / (self.gamma * norms)), 1)
        x_subvectors = tao.unsqueeze(2) * reshaped_x  # 形状为 (batch_size, num_subvectors, features_per_subvector)
        return x_subvectors.reshape(batch_size, total_features)  # 直接重塑为原始形状

    @classmethod
    def name(cls):
        return 'ProxCappedl1'

    def _split_vectors(self, x: torch.Tensor) -> list[torch.Tensor]:
        return torch.chunk(x, self.num_subvectors, dim=1)

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x_subvectors, dim=1)


class ProxLogSum(GroupProximalOperator):
    def __init__(self, n: int, gLen: int):
        super().__init__()
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)

    @classmethod
    def obj(cls, x: torch.Tensor, xtilde: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('Not implemented yet')

    def prox(self, x: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        batch_size, total_features = x.size()
        features_per_subvector = total_features // self.num_subvectors
        reshaped_x = x.reshape(batch_size, self.num_subvectors, features_per_subvector)
        norms = torch.norm(reshaped_x, p=2, dim=2)  #
        epsl = 4
        tao = torch.where(
            torch.logical_or(
                ((norms.detach() + epsl) ** 2 - 4 * v.detach() < 0) * (norms.detach() <= v.detach() / epsl),
                norms == 0),
            0,
            rx(norms, v, epsl) / norms)
        x_subvectors = tao.unsqueeze(2) * reshaped_x  # 形状为 (batch_size, num_subvectors, features_per_subvector)
        return x_subvectors.reshape(batch_size, total_features)  # 直接重塑为原始形状

    @classmethod
    def name(cls) -> str:
        return 'ProxLogSum'

    def _split_vectors(self, x: torch.Tensor) -> list[torch.Tensor]:
        return torch.chunk(x, self.num_subvectors, dim=1)

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x_subvectors, dim=1)


def rx(x, v, epsl):
    return 0.5 * (x - epsl) + 0.5 * torch.sqrt((x + epsl) ** 2 - 4 * v)


class ProxTransformedL1(GroupProximalOperator):
    def __init__(self, n: int, gLen: int, a: float):
        super().__init__()
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)
        self.a = a
        assert self.a > 0, ValueError('a must be greater than 0')

    def prox(self, x: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        batch_size, total_features = x.size()
        features_per_subvector = total_features // self.num_subvectors
        reshaped_x = x.reshape(batch_size, self.num_subvectors, features_per_subvector)
        norms = torch.norm(reshaped_x, p=2, dim=2)  # 形状为 (batch_size, num_subvectors)
        a = self.a
        t = torch.where(v.detach() <= a ** 2 / (2 * (a + 1)), v.detach() * (a + 1) / a,
                        torch.sqrt(2 * v.detach() * (a + 1)) - a / 2)
        # v=torch.where(v.detach()>0.3,0.3,v)
        tao = torch.where(
            norms.detach() <= t.detach(), 0,
            # norms.detach()<=t.detach(),0,
            ((2 / 3) * (a + norms) *
             torch.cos(
                 torch.arccos(
                     torch.clip((1 - 27 * v * a * (a + 1) / (2 * (a + norms) ** 3)),
                                min=torch.tensor(-1).cuda(),
                                max=torch.tensor(1).cuda()
                                )) / 3
                 #    1-(27*v)*a*(a+1)/(2*(a+norms)**3)/3)
             ) - 2 * a / 3 + norms / 3) / norms)
        x_subvectors = tao.unsqueeze(2) * reshaped_x  # 形状为 (batch_size, num_subvectors, features_per_subvector)
        return x_subvectors.reshape(batch_size, total_features)  # 直接重塑为原始形状

    @classmethod
    def name(cls):
        return 'ProxTransformedL1'

    def _split_vectors(self, x: torch.Tensor) -> list[torch.Tensor]:
        return torch.chunk(x, self.num_subvectors, dim=1)

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x_subvectors, dim=1)


class ProxCappedL1over2(GroupProximalOperator):
    def __init__(self, n: int, gLen: int, gamma: float):
        super().__init__()
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)
        self.gamma = gamma
        assert self.gamma > 0, ValueError('gamma must be greater than 0')

    @classmethod
    def prox_l1over2(cls, x, threshold):  # succeed
        condition = 1.5 * threshold ** (2.0 / 3.0)
        return torch.where(torch.abs(x) <= condition, 0.0,
                           (4.0 / 3.0) * x * (torch.cos((1.0 / 3.0) * torch.arccos(
                               torch.clip(-3 ** 1.5 / 4 * threshold * (torch.abs(x) ** -1.5),
                                          min=torch.tensor(-1.0).cuda(),
                                          max=torch.tensor(1.0).cuda()
                                          ))) ** 2)
                           )

    @classmethod
    def obj(cls, x: torch.Tensor, xtilde: torch.Tensor, v: torch.Tensor, gamma) -> torch.Tensor:
        f = (x / gamma) ** 0.5
        return 0.5*(x - xtilde) ** 2 + v * torch.where(f <= 1, f, 1)

    def prox(self, x: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        batch_size, total_features = x.size()
        features_per_subvector = total_features // self.num_subvectors
        reshaped_x = x.reshape(batch_size, self.num_subvectors, features_per_subvector)
        norms = torch.norm(reshaped_x, p=2, dim=2)  # 形状为 (batch_size, num_subvectors)
        gamma = torch.tensor(self.gamma, requires_grad=False)
        con = self.prox_l1over2(norms, v / (gamma) ** 0.5)
        u1_star = torch.where(con.detach() <= gamma, con, gamma)
        u2_star = torch.where(norms.detach() < gamma, gamma, norms)
        with torch.no_grad():
            condition = self.obj(u1_star, norms, v=v, gamma=self.gamma).detach() <= self.obj(u2_star, norms, v=v,
                                                                                             gamma=self.gamma).detach()
        tao = torch.where(condition.detach(),
                          u1_star / (norms),
                          u2_star / (norms))
        # with torch.no_grad():
        #     condition = self.obj(u1_star,norms,v=v,gamma=self.gamma).detach()<= self.obj(u2_star,norms,v=v,gamma=self.gamma).detach()
        # tao = u1_star/(norms+1e-8)
        x_subvectors = tao.unsqueeze(2) * reshaped_x  # 形状为 (batch_size, num_subvectors, features_per_subvector)
        return x_subvectors.reshape(batch_size, total_features)  # 直接重塑为原始形状

    @classmethod
    def name(cls):
        return 'ProxCappedL1over2'

    def _split_vectors(self, x: torch.Tensor) -> list[torch.Tensor]:
        return torch.chunk(x, self.num_subvectors, dim=1)

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x_subvectors, dim=1)


import torch.nn.functional as F


def shrink(x, theta):
    return x.sign() * F.relu(x.abs() - theta)


class ProxCappedTransformedL1(GroupProximalOperator):
    def __init__(self, n: int, gLen: int, a: float):
        super().__init__()
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)
        self.a = a
        self.tl1 = ProxTransformedL1(n, gLen, a)
        assert self.a > 0, ValueError('gamma must be greater than 0')

    def prox(self, x, lamb):
        batch_size, total_features = x.size()
        features_per_subvector = total_features // self.num_subvectors
        reshaped_x = x.reshape(batch_size, self.num_subvectors, features_per_subvector)
        norms = torch.norm(reshaped_x, p=2, dim=2)  # 形状为 (batch_size, num_subvectors)
        a = self.a
        # t = torch.where(lamb.detach() <= a ** 2 / (2 * (a + 1)), lamb.detach() * (a + 1) / a,
        #                 torch.sqrt(2 * lamb.detach() * (a + 1)) - a / 2)
        v = 1
        fv = (a + 1) * v / (a + v)

        px = self.tl1.prox(norms, v=lamb / fv)
        fpx = (a + 1) * px / (a + px)
        absx = norms
        # cappel l1
        # fv = v
        # px = shrink(absx, lamb / fv)
        # # phistar = lamb / v

        # cappel l1/2
        # fv=v**0.5
        # px = l1over2(absx, lamb/v)

        # fv=v**(2/3)
        # px = l2_2over3(absx, lamb/fv)
        # phistar =  (8 * (lamb/fv ** 3) / 729)

        # phistar = 1.5 * ( lamb/fv) ** (2.0 / 3.0)
        #  con0 = absx < t
        con1_1 = torch.logical_and(px <= absx, torch.logical_and(v > absx, absx <= (v / 2 + lamb / 2 + px / 2)))
        con1_2 = torch.logical_and(px <= absx, torch.logical_and(absx < v, absx > (v / 2 + lamb / 2 + px / 2)))
        con2_1 = torch.logical_and(px <= v,
                                   torch.logical_and(v < absx, absx <= torch.sqrt((2 * lamb * (1 - fpx / fv))) + px))
        #  con2_2 = torch.logical_and(px <= v, torch.logical_and(v < absx, absx > torch.sqrt((2 * lamb * (1 - fpx / fv))) + px))

        # ####################org
        # con2_1 = torch.logical_and(px <= v,
        #                            torch.logical_and(v < absx, absx <= torch.sqrt((2 * lamb * (1 - px / fv))) + px))
        # con2_2 = torch.logical_and(px <= v,
        #                            torch.logical_and(v < absx, absx > torch.sqrt((2 * lamb * (1 - px / fv))) + px))
        # con3 = v<px<absx

        # tao=torch.where(
        #     con0, 0,
        #     torch.where(con1_1, px/norms,
        #                 torch.where(con1_2, 1, torch.where(con2_1,  px/norms, 1)
        #                             )
        #                 )
        # )
        tao = torch.where(con1_1, px / norms,
                          torch.where(con1_2, 1, torch.where(con2_1, px / norms, 1)
                                      )
                          )

        x_subvectors = tao.unsqueeze(2) * reshaped_x  # 形状为 (batch_size, num_subvectors, features_per_subvector)
        return x_subvectors.reshape(batch_size, total_features)  # 直接重塑为原始形状

    @classmethod
    def name(cls):
        return 'ProxCappedTransformedL1'

    def _split_vectors(self, x: torch.Tensor) -> list[torch.Tensor]:
        return torch.chunk(x, self.num_subvectors, dim=1)

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x_subvectors, dim=1)


class ProxArctan(GroupProximalOperator):
    def __init__(self, n: int, gLen: int, c: float):
        super().__init__()
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)
        self.c = c

    def prox(self, x: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        batch_size, total_features = x.size()
        features_per_subvector = total_features // self.num_subvectors
        reshaped_x = x.reshape(batch_size, self.num_subvectors, features_per_subvector)
        norms = torch.norm(reshaped_x, p=2, dim=2)  # 形状为 (batch_size, num_subvectors)
        c = self.c
        p_absx = 1 / (3 * c ** 2) - (norms ** 2) / 9
        q_absx = v / (4 * c) - norms / (3 * c ** 2) - norms ** 3 / 27
        r = torch.sign(q_absx) * torch.sqrt(torch.abs(p_absx))
        tao = torch.where(norms <= self.c * v.detach() / 2, 0.0,
                          torch.where(
                              p_absx.detach() <= 0,
                              -2 * r * torch.cosh(
                                  torch.arccosh(torch.clip(q_absx / (r ** 3), min=1.001, max=10000)) / 3) + norms / 3,
                              -2 * r * torch.sinh(torch.arcsinh(q_absx / (r ** 3)) / 3) + norms / 3
                          ) / (norms)
                          )
        # 形状为 (batch_size, num_subvectors)
        x_subvectors = tao.unsqueeze(2) * reshaped_x  # 形状为 (batch_size, num_subvectors, features_per_subvector)

        return x_subvectors.reshape(batch_size, total_features)  # 直接重塑为原始形状

    @classmethod
    def name(cls) -> str:
        return 'ProxArctan'

    def _split_vectors(self, x: torch.Tensor) -> list[torch.Tensor]:
        return torch.chunk(x, self.num_subvectors, dim=1)

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x_subvectors, dim=1)


class ProxL1_1over2(GroupProximalOperator):
    def __init__(self, n: int, gLen: int):
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)


    def prox(self, x: torch.Tensor, v, *args, **kwargs) -> torch.Tensor:

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

        par = partial(self.L1_1over2_part.prox, nu=v_reshape, con1=con1, con2=con2, con4=con4, s_arange=s_arange)
        return torch.concatenate(list(map(par, x_subvectors))).reshape(batch_size,
                                                                       total_features)

    def _split_vectors(self, x: torch.Tensor) -> list[torch.Tensor]:
        return torch.chunk(x, self.num_subvectors, dim=1)

    @classmethod

    def name(cls):
        return 'ProxL1_1over2'

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError('You dont need to call this method')

    class L1_1over2_part:

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
            y_l1_array = y.cumsum(dim=0)
            cs_array = cls._calculate_cs(y_l1_array, nu, s_arange)
            new_yl1 = torch.cat((y[1:], torch.tensor([-1], device=y.device)))
            res = (y_l1_array >= con4) & (y_l1_array >= tmp) & (y > cs_array) & (
                    cs_array >= new_yl1)
            return [torch.relu(y - cs_array[ind]) for ind in torch.nonzero(res.flatten(), as_tuple=True)[
                0]]

        @staticmethod
        def _calculate_cs(y_l1_array: torch.Tensor, nu: torch.Tensor, s) -> torch.Tensor:
            return (math.sqrt(3.0) * nu) / (
                    4 * torch.sqrt(y_l1_array) * torch.cos(
                (1 / 3) * torch.arccos(
                    torch.clip(-3 * math.sqrt(3.0) * nu * s * torch.pow(y_l1_array, -3 / 2) / 4, min=-1, max=1))))

        @staticmethod
        def _calculate_condition3(v, q, s):
            return (2 - q) * (v * q * s / ((q - 1) ** (1 - q))) ** (1 / (2 - q))


class ProxL1_2over3(GroupProximalOperator):
    def __init__(self, n: int, gLen: int):
        self.n: int = n
        self.gLen: int = gLen
        self.num_subvectors: int = self.get_num_subvectors(n, gLen)

    def prox(self, x: torch.Tensor, v, *args, **kwargs) -> torch.Tensor:

        batch_size, total_features = x.size()

        features_per_subvector = total_features // self.num_subvectors

        x_subvectors = torch.chunk(x.reshape(-1), batch_size * self.num_subvectors, dim=-1)

        con1 = 4*(2/9)**(0.75)*v.detach()**(0.75)
        con2 = 2*(2*v.detach()/3)**(0.75)

        s_arange = torch.arange(1, features_per_subvector + 1, device=x.device)
        v_reshape = v.reshape(-1)
        con4 = ((4/3)*(v.detach()**0.75)*(2**0.75)/(3**0.5))*s_arange ** (0.75)

        par = partial(self.L1_2over3_part.prox, nu=v_reshape, con1=con1, con2=con2, con4=con4, s_arange=s_arange)
        return torch.concatenate(list(map(par, x_subvectors))).reshape(batch_size,total_features)

    def _split_vectors(self, x: torch.Tensor) -> list[torch.Tensor]:
        return torch.chunk(x, self.num_subvectors, dim=1)

    @classmethod
    def name(cls):
        return 'ProxL1_2over3'

    @staticmethod
    def _concentrate_vectors(x_subvectors: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError('You dont need to call this method')

    class L1_2over3_part:

        @staticmethod
        def obj(y: torch.Tensor, ytilde: torch.Tensor, nu: float) -> torch.Tensor:
            return nu * torch.sqrt(ytilde.sum()) + 0.5 * torch.norm(y - ytilde) ** 2

        @classmethod
        def prox(cls, x: torch.Tensor, nu: torch.Tensor, con1, con2, con4, s_arange) -> torch.Tensor:

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
                    return cls._rearrange_org(s_list[torch.argmin(torch.tensor(Js_list))], sorted_indices, sign_x)


        @staticmethod
        def _rearrange_org(ytilde: torch.Tensor, sorted_indices, sign: torch.Tensor) -> torch.Tensor:
            return ytilde[sorted_indices] * sign

        @classmethod
        def _loop_search_s(cls, y: torch.Tensor, nu: torch.Tensor, con4, s_arange) -> list[torch.Tensor]:
            tmp = torch.real(cls._calculate_condition3(v=nu, q=2/3, s=s_arange).reshape(-1))
            y_l1_array = y.cumsum(dim=0)
            cs_array = cls._calculate_cs(y_l1_array, nu, s_arange)
            new_yl1 = torch.cat((y[1:], torch.tensor([-1], device=y.device)))
            res = (y_l1_array >= con4) & (y_l1_array >= tmp) & (y > cs_array) & (
                    cs_array >= new_yl1)
            return [torch.relu(y - cs_array[ind]) for ind in torch.nonzero(res.flatten(), as_tuple=True)[
                0]]

        @staticmethod
        def _calculate_cs(y_l1_array: torch.Tensor, nu: torch.Tensor, s) -> torch.Tensor:
            t1 = y_l1_array**2/16
            t2 = torch.sqrt(torch.clip(t1**2-8*(nu**3)*(s**3),min=0.0,max=10000))
            alpha = (t1+t2)**(1/3) + (t1-t2)**(1/3)
            sq2alpha = torch.sqrt(2*alpha)
            return (4*nu/3)*(1/(sq2alpha+torch.sqrt(torch.clip(2*y_l1_array/sq2alpha-2*alpha,min=1e-3,max=100000))))

        @staticmethod
        def _calculate_condition3(v, q, s):
            return (2 - q) * (v * q * s / ((q - 1) ** (1 - q))) ** (1 / (2 - q))

