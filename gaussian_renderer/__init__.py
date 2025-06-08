#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_normalize
from utils.rigid_utils import from_homogenous, to_homogenous
from utils.graphics_utils import normal_from_depth_image



def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def render_normal(viewpoint_cam, depth, alpha):
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()

    normal_ref = normal_from_depth_image(depth, intrinsic_matrix.to(depth.device), extrinsic_matrix.to(depth.device))
    normal_ref = normal_ref
    normal_ref = normal_ref.permute(2,0,1)

    return normal_ref

def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, d_xyz, d_rotation, d_scaling, d_reflvec, iteration, opt,
           scaling_modifier=1.0, override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)



    means3D = pc.get_xyz + d_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = torch.abs(pc.get_scaling + d_scaling)
        rotations = torch.nn.functional.normalize(pc.get_rotation + d_rotation)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if iteration >= opt.warm_up + 3000:
        gb_pos = pc.get_xyz + d_xyz # (N, 3) 
        view_pos = viewpoint_camera.camera_center.repeat(pc.get_opacity.shape[0], 1) # (N, 3) 
        d_viewdir_normalized = safe_normalize(view_pos - gb_pos)
        deform_normal, _ = pc.get_normal(pc.get_scaling, pc.get_rotation, d_scaling, d_rotation, d_viewdir_normalized) # (N, 3) 
    
    if colors_precomp is None:
        if iteration >= opt.warm_up2:
            diffuse   = pc.get_diffuse 
            specular  = pc.get_specular 
            roughness = pc.get_roughness 
            color = pc.brdf_mlp.shade(gb_pos[None, None, ...].detach(), deform_normal[None, None, ...], d_reflvec[None, None, ...], diffuse[None, None, ...], specular[None, None, ...], roughness[None, None, ...], view_pos[None, None, ...])
            colors_precomp = color.squeeze() 
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        }

    if iteration >= opt.warm_up + 3000:
        p_hom = torch.cat([pc.get_xyz + d_xyz, torch.ones_like(pc.get_xyz[...,:1])], -1).unsqueeze(-1)
        p_view = torch.matmul(viewpoint_camera.world_view_transform.transpose(0,1), p_hom)
        p_view = p_view[...,:3,:]
        depth = p_view.squeeze()[...,2:3]
        depth = depth.repeat(1,3)
        render_extras = {"depth": depth}
        normal_normed = 0.5*deform_normal + 0.5  
        render_extras.update({"normal": normal_normed})
    
        out_extras = {}
        for k in render_extras.keys():
            if render_extras[k] is None: continue
            image = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = None,
                colors_precomp = render_extras[k],
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)[0]
            out_extras[k] = image       
        for k in ["normal"]:
            if k in out_extras.keys():
                out_extras[k] = (out_extras[k] - 0.5) * 2. 
                torch.nn.functional.normalize(out_extras[k], p=2, dim=0)
        
        raster_settings_alpha = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor([0,0,0], dtype=torch.float32, device="cuda"),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug,
        )
        rasterizer_alpha = GaussianRasterizer(raster_settings=raster_settings_alpha)
        alpha = torch.ones_like(means3D) 
        out_extras["alpha"] =  rasterizer_alpha(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = alpha,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)[0]
        
        out_extras["normal_ref"] = render_normal(viewpoint_cam=viewpoint_camera, depth=out_extras['depth'][0], alpha=out_extras['alpha'][0])
        out.update(out_extras)
    return out
