import glob
import os
import subprocess
import sys
import numpy as np
import trimesh
from binvox import binvox
import binvox_rw
#help(binvox)

vox_res = 32
dir_path = r"/Volumes/ROCKET-nime/ML_data/MN10/ModelNet10/"
dataset_path = r"/Volumes/ROCKET-nime/ML_data/MN10/mn10_binvox"

"""for dir in os.listdir(dir_path):
    if not dir.startswith("._"):
        #os.mkdir(os.path.join(dataset_path, dir))
        os.mkdir(os.path.join(dataset_path,dir,"train"))
        os.mkdir(os.path.join(dataset_path,dir,"test"))

        print(dir)
"""
paths = glob.glob(os.path.join(dir_path + "*","*","*.off"))
#paths = paths[1:2]
print("number of data:", len(paths))

def parse_OFF(file_path):
                # def parse_OFF(text,label):
                file_path = file_path
                if not file_path.endswith(".off"):
                    pass

                text = open(file_path, "r")
                text = text.readlines()

                #FIX OFF FORMAT
                if text[0].strip().startswith("OFF") and text[0].strip().endswith("OFF"):
                    pass
                else:
                    other = text[0][3:]
                    text[0] = "OFF"
                    text.insert(1, other)


                vertices = []
                faces = []
                for i, line in enumerate(text):
                    line = line.split(" ")
                    if i == 0:
                        pass

                    elif i == 1:
                        n_vertices = int(line[0])
                        n_faces = int(line[1])

                    elif i <= int(n_vertices) + 1:
                        x = line[0]
                        y = line[1]
                        z = line[2]
                        p = [x, y, z]
                        vertices.append(p)

                    else:
                        # check for triangular faces
                        n_f = int(line[0])

                        if n_f == 3:
                            f = []
                            for j in range(n_f):
                                f.append(line[j + 1])
                            faces.append(f)
                        else:
                            break
                    continue

                vertices = np.asarray(vertices).astype('float32')
                faces = np.asarray(faces).astype('float32')
                return vertices, faces

for file_path in paths:
    print()
    file_name = file_path.split(os.sep)[-1]
    vertices, faces = parse_OFF(file_path)
    mesh = trimesh.base.Trimesh(vertices, faces)

    bounds = mesh.bounds
    l, w, d = bounds[1][0] - bounds[0][0], bounds[1][1] - bounds[0][1], bounds[1][2] - bounds[0][2]

    vox = mesh.voxelized(max(l, w, d) / vox_res).matrix
    bin_vox = trimesh.exchange.binvox.Binvox(mesh, vox.shape, translate=None, scale=None)
    bin_vox = binvox.Binvox(vox, dims=vox.shape, axis_order="xyz")


    def save_binvox(filename, data):
        dims = data.shape
        translate = [0.0, 0.0, 0.0]
        model = binvox_rw.Voxels(data, dims, translate, 1.0, 'xyz')
        print(model)
        with open(filename, 'w') as f:
            model.write(f)

    out_path = os.path.join(dataset_path,"/".join(file_path.split(os.sep)[-3:]))
    save_binvox(out_path[:-4] + ".binvox", bin_vox.data)

