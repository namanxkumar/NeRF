import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, List, Union, Callable
import torch


class Dataloader():
    def __init__(self):
        if not os.path.exists('tiny_nerf_data.npz'):
            print("Dataset not found")
            exit()

        self.data = np.load('tiny_nerf_data.npz') # load the dataset from https://cseweb.ucsd.edu//~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz
        self.images = self.data['images']
        self.poses = self.data['poses']
        self.focal = self.data['focal']

        self.height, self.width = self.images.shape[1:3]
        self.near, self.far = 2.0, 6.0

        self.training_length = 100

    def show_image(self, image_index: int):
        image, pose = self.images[image_index], self.poses[image_index] # get the image and the pose at the given index
        plt.imshow(image)
        print('Pose')
        print(pose)
        plt.show()

    def show_cameras(self):
        rotations = np.stack([np.sum([0, 0, -1]*pose[:3, :3], -1) for pose in self.poses]) # get rotation of the camera, i.e. the direction of the z-vector, since it points along the camera axis
        origins = self.poses[:, :3, -1] # get the origin of the camera, i.e. the translation vector
        figure = plt.figure(figsize=(12, 8)).add_subplot(projection='3d') # create a 3D plot
        _ = figure.quiver(origins[:, 0], origins[:, 1], origins[:, 2], rotations[:, 0], rotations[:, 1], rotations[:, 2], length=0.5, normalize=True) # plot the camera positions and directions
        figure.set_xlabel('X')
        figure.set_ylabel('Y')
        figure.set_zlabel('Z')
        plt.show()
    
    def get_rays(self, image_index: int):
        x_coordinates, y_coordinates = torch.meshgrid(torch.arange(self.width, dtype=torch.float32), torch.arange(100, dtype=torch.float32), indexing='xy') # make a grid of x and y coordinates of the pixels
        directions = torch.stack([(x_coordinates-self.width/2)/self.focal, -(y_coordinates-self.height/2)/self.focal, -torch.ones_like(x_coordinates)], dim=-1) # calculate the direction of the rays, this matrix multiplied with distance form optical center, w, will give the u, v, w coordinates in world space
        # Shape of directions: (width, height, 3)

        rays_directions = torch.sum(directions[..., None, :] * self.poses[image_index, :3, :3], -1) # multiply the direction of the rays with the rotation matrix of the camera to get the direction of the rays in world space
        rays_origin = self.poses[image_index, :3, -1] # get the origin of the camera in world space

        return rays_origin, rays_directions

if __name__ == '__main__':
    dataloader = Dataloader()
    dataloader.show_image(10)
    dataloader.show_cameras()
