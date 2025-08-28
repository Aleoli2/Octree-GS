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
from torch import nn
from scene import Scene
from gaussian_renderer import render, prefilter_voxel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.graphics_utils import getProjectionMatrix, focal2fov
import time
import numpy as np
import cv2


class View(nn.Module):
    def __init__(self, R, T, focal_x, focal_y, width, height, data_device = "cuda" ):
        super(View, self).__init__()


        self.R = R
        self.T = T
        self.focal_y = focal_x
        self.focal_x = focal_y
        self.image_width = width
        self.image_height = height
        self.FoVx = focal2fov(focal_x,width)
        self.FoVy = focal2fov(focal_y,height)

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.zfar = 100.0
        self.znear = 0.01
        
        self.getWorldView(R,T)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def getWorldView(self, R, T):
        self.world_view_transform = np.zeros((4, 4))
        self.world_view_transform[3, 3] = 1.0
        t_inv = np.matmul(-R.transpose(),T)
        self.world_view_transform[:3,:3]=R.transpose()
        self.world_view_transform[:3,3] = t_inv
        self.world_view_transform = torch.tensor(np.float32(self.world_view_transform)).transpose(0, 1).cuda()
        
class Renderer():
    def __init__(self, args=None):
        # Set up command line argument parser
        parser = ArgumentParser(description="Testing script parameters")
        
        self.model = ModelParams(parser, sentinel=True)
        self.pipeline = PipelineParams(parser)
        parser.add_argument("--iteration", default=-1, type=int)
        args = get_combined_args(parser, args)

        camera_model_path = args.camera_model
        with open(camera_model_path, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                data = line.split(" ")
                break
        self.camera_model = { "fl_x": float(data[4]),
        "fl_y": float(data[5]),
        "w": float(data[2]),
        "h": float(data[3])}

        dataset = self.model.extract(args)
        self.pipeline = self.pipeline.extract(args)
        self.gaussians = GaussianModel(
            dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim, 
            dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level, 
            dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend
        )
        #Load model
        self.iteration=args.iteration
        scene = Scene(dataset, self.gaussians, load_iteration=args.iteration, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    def render_view(self, rotation, translation):
        view = View(rotation, translation, self.camera_model["fl_x"], self.camera_model["fl_y"], self.camera_model["w"], self.camera_model["h"])
        self.gaussians.set_anchor_mask(view.camera_center,self.iteration, resolution_scale=1.0)
        voxel_visible_mask = prefilter_voxel(view, self.gaussians, self.pipeline, self.background)
        rendering = render(view, self.gaussians, self.pipeline, self.background, visible_mask=voxel_visible_mask, ape_code=10)["render"]
        rendering = rendering[:3, :, :]
        image = rendering.data.cpu().numpy()
        image *= 255
        return image.astype(np.uint8)

if __name__ == "__main__":
    renderer = Renderer()
    R = np.array([[-0.9439490687833039, -0.012658517779477973, 0.32984832494763394], [-0.32913601383730196, -0.039867014542148096, -0.9434405681052662], [0.025092627172631866, -0.9991248085595327, 0.03346605716920529]])
    T = np.array([-12.574966256566968, 38.94981550209616, -0.06905938699614564])
    start=time.clock_gettime_ns(time.CLOCK_THREAD_CPUTIME_ID)
    for _ in range(30):
        image=renderer.render_view(R, T)
    end=time.clock_gettime_ns(time.CLOCK_THREAD_CPUTIME_ID)
    print("Mean rendering time (ms): {} FPS: {}".format((end-start)*10**-6/30,(1/((end-start)*10**-9)*30)))
    image = image.swapaxes(0, 2).swapaxes(0, 1)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test.png", image)