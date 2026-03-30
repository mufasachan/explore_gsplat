from typing import Literal
import torch
from torch import Tensor
from torch.nn import ParameterDict
from gsplat.rendering import rasterization


def rasterize_splats(
    splats: ParameterDict,
    camtoworlds: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
) -> Tensor:
    means = splats["means"]  # [N, 3]
    quats = splats["quats"]  # [N, 4]
    scales = torch.exp(splats["scales"])  # [N, 3]
    opacities = torch.sigmoid(splats["opacities"])  # [N,]
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, K, 3]

    render_colors, _, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=torch.linalg.inv(camtoworlds),  # [C=1, 4, 4]
        sh_degree=3,
        Ks=Ks,  # [C=1, 3, 3]
        width=width,
        height=height,
        packed=True,
        rasterize_mode="classic",
        distributed=False,
        camera_model="pinhole",
    )

    return render_colors
