# NeRF
 From scratch implementation of the original NeRF paper in PyTorch

 - [x] Dataloader
    - [x] Load Data
    - [x] Get rays form pinhole camera model
- [x] Model
    - [x] Encoding
    - [x] Implementation
- [ ] Volume Rendering
    - [ ] Volume Integration
    - [ ] Heirarchical Volume Sampling
- [ ] Training
    - [ ] Forward Pass
    - [ ] Batching Camera Rays
    - [ ]

## How the get_rays() function works
1. A meshgrid is created with arrays of size height and width (in pixels) of the image as input. The meshgrid essentially creates 2 matrices, which taken element wise pairs from comes out to be cartesian coordinates such as (0,1), (0,2), ... (1, 3), (1, 4) ... and so on. If the meshgrid indexing is 'xy' then the coordinates are taken by looping over the height pixels first, i.e. (h1, w0), (h2, w0), (h3, w0) ... will be the first row and then width pixel is incremented. In case of 'ij' indexing, width pixels are taken first. We take 'xy' indexing.
2. Next, we create a stack from the meshgrid to make matrix of size (width, height, 3). Note that width is along the x-axis in the image plane, and height is along the -y axis. the optical axis is -z. The point where optical axis meets the image plane is taken as ther origin of image plane. So, to shift the coordinates to the new frame, we subtract w/2 and h/2 for each x and y coordinate respectively. z need not be shifted. Once shifted to the new origin (new coordinates x', y', z'(=z)), we can apply similar triangles to convert from image coordinates to world coordinates, where (see diagram) x'/f = u/w and y'/f = v/w. Hence we divide the x' and y' coordinates by the focal length f. This is done so that when the resultant matrix is multiplied by the distance from the origin on the optical axis w, the u and v coordinates in the world frame are yielded.
3. Finally, we multiply every (x'/f, y'/f, z') coordinate with the camera rotation matrix to rotate the optical axis to the camera axis. The camera displacement also provides the origin of the optical axis.

