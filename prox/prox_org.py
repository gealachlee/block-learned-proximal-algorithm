import torch
import warnings
warnings.warn('prox_org is abandand,please use prox_cl_acc instead')
sparsity = 8
gLen = 16
gNo = int(512 / gLen)


def proxl2_1(x: torch.Tensor, v: torch.Tensor):
    num_subvectors = 512 // 16
    x_subvectors = torch.chunk(x, num_subvectors, dim=1)

    # 提取每个子向量并计算norm
    norms = torch.stack([subvec.norm(dim=1, p=2) for subvec in x_subvectors], dim=0)
    norm_x = norms
    # 使用torch.where和广播操作
    tao = torch.where(norm_x <= v, 0.0, 1 - v / (norm_x))
    # # method1
    # x2=torch.cat(x_subvectors, dim=1).reshape(x.shape[0],64,16)
    # return(tao.T.unsqueeze(2) * x2).reshape(x2.shape[0], -1)

    # 此处不影响反向传播
    x_subvectors = [torch.mul(tao[i].reshape(-1, 1), x_subvectors[i]) for i in range(num_subvectors)]
    x = torch.cat(x_subvectors, dim=1)
    x=  torch.nan_to_num(x,0)
    return x


def operator(x: torch.Tensor, v: torch.Tensor):
    num_subvectors = 512 // 16
    x_subvectors = torch.chunk(x, num_subvectors, dim=1)

    # 提取每个子向量并计算norm
    norms = torch.stack([subvec.norm(dim=1, p=2) for subvec in x_subvectors], dim=0)
    norm_x = norms

    # 计算p值
    p = -((3 ** 1.5) / 4) * v * norm_x ** (-1.5)  # ?

    # 使用torch.where和广播操作
    tao = torch.where(norm_x <= 1.5 * v.detach() ** (2 / 3), 0.0,
                      (2 / 3) * (1.0 + torch.cos((2 / 3) * torch.arccos(p))))
    # method1
    # x2=torch.cat(x_subvectors, dim=1).reshape(x.shape[0],64,16)
    # return(tao.T.unsqueeze(2) * x2).reshape(x2.shape[0], -1)

    # method 2
    # for i in range(0, len(x[0]), 16):
    #     x[:, i:i + 16] = torch.mul(tao[int(i//16)].reshape(-1, 1), x_subvectors[i//16])
    # return x

    # method 3
    # 此处不影响反向传播
    x_subvectors = [torch.mul(tao[i].reshape(-1, 1), x_subvectors[i]) for i in range(num_subvectors)]
    x = torch.cat(x_subvectors, dim=1)
    return x


def proxl2_1over2(x: torch.Tensor, v: torch.Tensor):
    num_subvectors = 512 // 16
    x_subvectors = torch.chunk(x, num_subvectors, dim=1)

    # 提取每个子向量并计算norm
    norms = torch.stack([subvec.norm(dim=1, p=2) for subvec in x_subvectors], dim=0)
    norm_x = norms

    # 计算p值

    # 使用torch.where和广播操作
    break_point = 1.5 * v ** (2 / 3)
    tao = torch.where(norm_x <= break_point , 0.0, (2 / 3) * (1.0 + torch.cos((2 / 3) *
                                                  (torch.arccos(
                                                      torch.clip(-3 ** 1.5 / 4 * v * (norm_x ** -1.5),
                                                                 min=torch.tensor(-1 + 0.0001).cuda(),
                                                                 max=torch.tensor(1 - 0.0001).cuda()
                                                                 )
                                                  )
                                                  ))))
    x_subvectors = [torch.mul(tao[i].reshape(-1, 1), x_subvectors[i]) for i in range(num_subvectors)]
    x = torch.cat(x_subvectors, dim=1)
    return x


def l2_1over2(x: torch.Tensor, v: torch.Tensor):
    num_subvectors = 512 // 16
    x_subvectors = torch.chunk(x, num_subvectors, dim=1)

    # 提取每个子向量并计算norm
    norms = torch.stack([subvec.norm(dim=1, p=2) for subvec in x_subvectors], dim=0)
    epsl = 0.1
    condition = 1.5 * v ** (2 / 3)
    # target_point = 0.6666 * (1 + torch.cos(0.6666 * torch.arccos(-3 ** 1.5 / 4 * v * (condition) ** -1.5)))
    tao = torch.where(
        norms <= condition ,
        0, torch.where(
            norms >= condition + epsl,
            (2/3) * (1 + torch.cos( (2/3) * torch.arccos(-3 ** 1.5 / 4 * v * (norms) ** -1.5))),
            (2/3) * (1 + torch.cos( (2/3) * torch.arccos(-3 ** 1.5 / 4 * v * (norms ) ** -1.5))) * epsl / (2)
        ))

    # 此处不影响反向传播
    # print(v)
    x_subvectors = [torch.mul(tao[i].reshape(-1, 1), x_subvectors[i]) for i in range(num_subvectors)]
    x = torch.cat(x_subvectors, dim=1)
    return x


def proxl2_2over3(x: torch.Tensor, v: torch.Tensor):
    # num_subvectors = 512  # 逐个元素
    num_subvectors = 512 // 16
    x_subvectors = torch.chunk(x, num_subvectors, dim=1)

    # 提取每个子向量并计算norm
    norms = torch.stack([subvec.norm(dim=1, p=2) for subvec in x_subvectors], dim=0)
    norms = torch.where(
        (1 / 256 - (8 * (v.detach() ** 3) / (729*norms.detach() ** 4))) <= 0,
        torch.sqrt(torch.sqrt(8 * 256 * (v ** 3) / 729)) + 1e-12,
        norms)
    t = (1/ 16 + torch.sqrt( 1 / 256 - (8 * (v ** 3) / (729*norms ** 4)))) ** (1 / 3) + (
            1 / 16 - torch.sqrt(1 / 256 - (8 * (v ** 3) / (729*norms ** 4)))) ** (1 / 3)
    tao = torch.where(norms.detach() <= 2 * ((2 / 3) * v.detach() ** 0.75), 0.0,
                      (1.0 / 8.0)  * (torch.sqrt(2.0 * t) + (
                          torch.sqrt(2.0  / torch.sqrt(2.0 * t) - 2.0 * t))) ** 3)
    # 此处不影响反向传播
    x_subvectors = [torch.mul(tao[i].reshape(-1, 1), x_subvectors[i]) for i in range(num_subvectors)]
    x = torch.cat(x_subvectors, dim=1)
    return x

def shrink(x, theta):
    return x.sign() * torch.nn.functional.relu(x.abs() - theta)

def proxMCP(x: torch.Tensor, v: torch.Tensor,gamma:torch.Tensor):
    num_subvectors = 512 // 16
    x_subvectors = torch.chunk(x, num_subvectors, dim=1)

    # 提取每个子向量并计算norm
    norms = torch.stack([subvec.norm(dim=1, p=2) for subvec in x_subvectors], dim=0)

    # 此处不影响反向传播
    # for index,x_subvector in enumerate(x_subvectors):
    #     x_subvectors[index]=torch.where(norms <= gamma * v, (gamma / (gamma - 1)) * shrink(x_subvector, v),x_subvector)
    #
    # x = torch.cat(x_subvectors, dim=1)
    tao=torch.where((norms <= gamma * v), torch.nn.functional.relu(1 - v/norms) * gamma * (gamma - 1), 1)
    #x_subvectors[0] = torch.where((x_subvectors[0].norm(dim=1,p=2)<=gamma*v),(gamma / (gamma - 1)) * shrink(x_subvectors[0], v),x_subvectors[0])
    x_subvectors = [torch.mul(tao[i].reshape(-1, 1), x_subvectors[i]) for i in range(num_subvectors)]
    x = torch.cat(x_subvectors, dim=1)
    return x