import numpy as np
from scipy.spatial import Delaunay
import random
import os
import cv2

# This script take images and turn them into terrain/height map
# This is then save as .obj file that can be used for 3D software like blender or unreal engine
# This is basically a way to generate unlimited amount of terrains/height maps with high diversity for procedural 3D scene making
# In addition you can load 2D shapes that will define the 2d shape of the terrain,eight map (x,y) if you dont add this  the topographical surface/ terrain generated will be square.
#-----------------Input Parameters--------------------------------------------------------------------------

shape_dir = "shapes//"
    #r"/media/deadcrow/SSD_480GB/segment_anything/shapes_512//"  # Folder with 2d shapes (saved as binary map) use for the 2D shape of of the terrain (optional only use if square=False)
img_dir = "images//"
    #r"/media/deadcrow/SSD_480GB/segment_anything/sa_000719//"  # Images that will be used to generate the height map
out_dir = "output//"
    #"topological_square_100x100//"  # Path where object/terrain will be saved
map_size = 100 # Sized of the topological height map
smoothing = 7  # smoothing factor
square=False # ignore 2D shape and generate square  map
flat=False # ignore topology/height and generate flat shape
MaxObject=80000
#-----------------------------------------------------------------------------------------------------------

def generate_topological_surface(mp, obj_filename="topological_surface.obj"):
    """
    Generates a 3D topological surface from a topological map and saves it as a Wavefront OBJ file.

    The topological map (mp) is assumed to be a numpy array of shape (h, w, 3), where:
      - mp[:, :, 0] is a binary mask (1 where the surface is defined, 0 elsewhere),
      - mp[:, :, 1] gives the z value (height) at each (x, y),
      - mp[:, :, 2] is ignored in this case.

    The function uses 2D Delaunay triangulation over the (x, y) coordinates of the valid (masked) pixels
    and assigns each vertex its corresponding z value from mp[:, :, 1]. Triangles whose centroids fall outside
    the valid region are discarded.

    Parameters:
      mp           : numpy.ndarray of shape (h, w, 3)
      obj_filename : string specifying the OBJ file to be saved (default "topological_surface.obj")

    Returns:
      vertices : (N, 3) array of vertex coordinates (x, y, z)
      faces    : (M, 3) array of triangles (indices into vertices, zero-indexed)
    """
    h, w, _ = mp.shape
    # Create binary mask (True where surface exists)
    mask = mp[:, :, 0] > 0

    # Find pixel indices (row, column) where mask is True.
    indices = np.argwhere(mask)  # Each row is (i, j): (row, col)

    # For triangulation, convert to 2D points with coordinates (x, y)
    # We'll use x = column index and y = row index.
    points_2d = indices[:, [1, 0]].astype(float)

    # Build the vertex array (x, y, z) where z is taken from mp[:, :, 1].
    vertices = np.empty((points_2d.shape[0], 3), dtype=float)
    vertices[:, 0] = points_2d[:, 0] - w/2 # x coordinate
    vertices[:, 1] = points_2d[:, 1]  - h/2 # y coordinate
    # For each valid pixel, retrieve its z value.
    vertices[:, 2] = np.array([mp[i, j, 1] for i, j in indices])

    # Perform Delaunay triangulation on the set of 2D points.
    delaunay = Delaunay(points_2d)
    all_faces = delaunay.simplices  # Each face is a triple of indices into points_2d

    # Filter triangles: discard triangles whose centroid is outside the valid mask.
    valid_faces = []
    for face in all_faces:
        pts = points_2d[face]
        centroid = np.mean(pts, axis=0)
        # Round the centroid to get the corresponding pixel indices.
        j_cent = int(round(centroid[0]))
        i_cent = int(round(centroid[1]))
        if 0 <= i_cent < h and 0 <= j_cent < w and mask[i_cent, j_cent]:
            valid_faces.append(face)
    faces = np.array(valid_faces)

    # Write the resulting mesh to an OBJ file.
    with open(obj_filename, "w") as f:
        # Write vertices: each line has "v x y z"
        for v in vertices:
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        # Write faces: OBJ format uses 1-indexing.
        for face in faces:
            f.write("f {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))

    print("Saved topological surface as '{}'".format(obj_filename))
    return vertices, faces


###########################################################################################################################
def make_shape(shape_list,img_list,sz=100,square=False,flat=False,smoothing=7):
    if not square:
           shapemask=cv2.imread(random.choice(shape_list),0)
    else:
           shapemask=np.ones([300,300])
    if flat:
        zmap=np.zeros_like(shapemask)
    else:
        zmap=cv2.imread(random.choice(img_list)).mean(2)
    hz, wz = zmap.shape
    hs,ws=shapemask.shape
#**********************************************************
    zmap = cv2.resize(zmap.astype(np.uint8), (int(wz / 4), int(hz / 3)))
    zmap = cv2.blur(zmap / 2, (smoothing, smoothing))
    zmap = cv2.blur(zmap / 2, (smoothing, smoothing))
    hz, wz = zmap.shape
#******************************************************

    r= np.min([hz/hs,wz/ws])
    if r<1:
        shapemask=(cv2.resize(shapemask,(int(ws*r),int(hs*r)))>0)
        hs, ws = shapemask.shape

    x = np.random.randint(wz+1 - ws)
    y = np.random.randint(hz+1 - hs)
    zmap=zmap[y:y+hs,x:x+ws].astype(np.float32)
    zmap*=shapemask
    zmean=zmap.sum()/shapemask.sum()
    zmap=zmap.astype(np.float32)-zmean
    if zmap.max()-zmap.min()>0:
         zmap*=sz/3/(zmap.max()-zmap.min())
    shape3d=np.concatenate([shapemask[:,:,None].astype(np.float32),zmap[:,:,None].astype(np.float32)],axis=2)
    shape3d=cv2.resize(shape3d,(sz,sz))
    return  shape3d#.astype(np.uint8)

#########################################################################################################################
# Example usage:

if __name__ == "__main__":

    if not os.path.exists(out_dir): os.mkdir(out_dir)
    shape_list=[]
    for fl in os.listdir(shape_dir):
        if ".png" in fl:
           shape_list.append(shape_dir+"//"+fl)

    image_list = []
    for fl in os.listdir(img_dir):
        if ".jpg" in fl:
          image_list.append(img_dir + "//" + fl)
    for ii in range(len(image_list)):
         if ii>MaxObject: break
         mp=make_shape(shape_list, image_list[ii:ii+1],sz=map_size,square=square,flat=flat,smoothing=smoothing)
         generate_topological_surface(mp, obj_filename=out_dir+"//"+str(ii)+".obj")
