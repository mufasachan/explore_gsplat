import math
import torch
from torch import Tensor
from torch.nn import ParameterDict
from gsplat.rendering import (
    isect_offset_encode,
    isect_tiles,
    rasterization,
    fully_fused_projection,
)


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


def project_splats(
    splats: ParameterDict,
    camtoworlds: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    means = splats["means"]  # [N, 3]
    quats = splats["quats"]  # [N, 4]
    scales = torch.exp(splats["scales"])  # [N, 3]
    opacities = torch.sigmoid(splats["opacities"])  # [N,]
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, K, 3]
    viewmats = torch.linalg.inv(camtoworlds)
    sh_degree = 3  # default training configuration
    covars = None
    eps2d = 0.3  # default value, ensure minimal size for gaussians
    # min/max distance for plans, used for clipping means
    near_plane, far_plane = (1e-2, 1e10)
    radius_clip = 0.0  # skip gaussian with a radius below this, not used

    # guards extracted from the gsplat.rendering.rasterization
    N = means.shape[-2]
    C = viewmats.shape[-3]
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert opacities.shape == (N,), opacities.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape
    assert (
        colors.dim() == 3 and colors.shape[:-2] == (N,) and colors.shape[-1] == 3
    ) or (
        colors.dim() == 4 and colors.shape[:-2] == (C, N) and colors.shape[-1] == 3
    ), colors.shape
    assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape

    (  # pyright: ignore[reportAssignmentType]
        _,  # batch_ids, all samples are in the same batch
        _,  # camera_ids, all samples have the same camera
        gaussian_ids,  # splats used ids
        radii,
        means2d,
        depths,
        conics,
        _,  # compensations None because no antialiasing
    ) = fully_fused_projection(
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        packed=True,  # trade memory(-) for time(+)
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        sparse_grad=False,  # better memory of the backword, no backward here
        calc_compensations=False,  # no antialiasing
        camera_model="pinhole",
        opacities=opacities,  # use opacities to compute a tigher bound for radii.
    )
    opacities = opacities[gaussian_ids]  # [nnz]

    return (gaussian_ids, radii, means2d, depths, conics)


def get_intersectiong_tiles(
    means2d: Tensor,
    radii: Tensor,
    depths: Tensor,
    gaussian_ids: Tensor,
    width: int,
    height: int,
    tile_size: int = 16,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # [n] Only one cameras, regenarate image_ids
    image_ids = torch.zeros_like(gaussian_ids)
    # C=I=1 (one camera, single image)
    C, I = 1, 1
    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        segmented=False,  # faster optimization, consumes more memory, more complex
        packed=True,  # trade memory(-) for time(+)
        n_images=I,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
    )
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    isect_offsets = isect_offsets.squeeze(0)

    return tiles_per_gauss, isect_ids, flatten_ids, isect_offsets
