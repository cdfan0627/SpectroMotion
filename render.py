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
from scene import Scene, DeformModel, DeformEnvModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state, safe_normalize, reflect
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel




def render_set(model_path, opt, name, iteration, views, gaussians, pipeline, background, deform, deform_env):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        view.load2device()
        fid = view.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
        gb_pos = gaussians.get_xyz + d_xyz 
        view_pos = view.camera_center.repeat(gaussians.get_opacity.shape[0], 1) 
        d_viewdir_normalized = safe_normalize(view_pos - gb_pos)
        normal, deform_delta_normal = gaussians.get_normal(gaussians.get_scaling, gaussians.get_rotation, d_scaling, d_rotation, d_viewdir_normalized)
        reflvec = safe_normalize(reflect(d_viewdir_normalized, normal))
        d_reflvec = deform_env.step(reflvec.detach(), time_input)        
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, d_reflvec, iteration, opt)
        rendering = results["render"]

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))



            

def render_sets(dataset: ModelParams, opt: OptimizationParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.brdf_envmap_res)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        deform = DeformModel()
        deform.load_weights(dataset.model_path)
        deform_env = DeformEnvModel(dataset.t_multires)
        deform_env.load_weights(dataset.model_path)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


        if not skip_train:
            render_set(dataset.model_path, opt, "train", scene.loaded_iter,
                        scene.getTrainCameras(), gaussians, pipeline,
                        background, deform, deform_env)

        if not skip_test:
            render_set(dataset.model_path, opt, "test", scene.loaded_iter,
                        scene.getTestCameras(), gaussians, pipeline,
                        background, deform, deform_env)
            

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), op.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
