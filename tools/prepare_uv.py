import os
import numpy as np
from tqdm import tqdm
from psbody.mesh import Mesh
from termcolor import colored
import argparse
import cv2


def myprint(cmd, level):
    color = {'run': 'blue', 'info': 'green', 'warn': 'yellow', 'error': 'red'}[level]
    print(colored(cmd, color))


def log(text):
    myprint(text, 'info')


def load_obj(path):
    """
    Load .obj file (UV model)
    """
    model = {}
    pts = []
    tex = []
    faces = []

    with open(path) as file:
        while True:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                pts.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "vt":
                tex.append((float(strs[1]), float(strs[2])))

    uv = np.zeros([len(pts), 2], dtype=np.float32)
    with open(path) as file:
        while True:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "f":
                face = (int(strs[1].split("/")[0]) - 1,
                        int(strs[2].split("/")[0]) - 1,
                        int(strs[4].split("/")[0]) - 1)
                texcoord = (int(strs[1].split("/")[1]) - 1,
                            int(strs[2].split("/")[1]) - 1,
                            int(strs[4].split("/")[1]) - 1)
                faces.append(face)
                for i in range(3):
                    uv[face[i]] = tex[texcoord[i]]

        model['pts'] = np.array(pts)
        model['faces'] = np.array(faces)
        model['uv'] = uv

    return model


def read_pickle(pkl_path):
    import pickle
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()


def barycentric_interpolation(val, coords):
    """
    Perform barycentric interpolation.
    :param val: Vertex attributes matrix (vertices x d)
    :param coords: Barycentric coordinates (vertices x 3)
    :return: Interpolated values
    """
    t = val * coords[..., np.newaxis]
    ret = t.sum(axis=1)
    return ret


def get_grid_points(vertices, padding=0.05, voxel_size=0.025):
    """
    Create a 3D grid around vertices
    """
    min_xyz = np.min(vertices, axis=0) - padding
    max_xyz = np.max(vertices, axis=0) + padding
    x = np.arange(min_xyz[0], max_xyz[0] + voxel_size, voxel_size)
    y = np.arange(min_xyz[1], max_xyz[1] + voxel_size, voxel_size)
    z = np.arange(min_xyz[2], max_xyz[2] + voxel_size, voxel_size)
    pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
    return pts


def get_uv_per_frame(frame, params_dir, vertices_dir, output_dir, uv_model, smpl_model_path):
    """
    Generate UV mapping for a specific frame in shape (d, h, w, 2),
    using vertices transformed into SMPL coordinate system.
    """
    param_path = os.path.join(params_dir, f'{frame}.npy')
    vertices_path = os.path.join(vertices_dir, f'{frame}.npy')

    # Load parameters and vertices
    params = np.load(param_path, allow_pickle=True).item()
    vertices = np.load(vertices_path)

    # Extract SMPL global transformation parameters
    Rh = params['Rh'].astype(np.float32)  # Global rotation vector
    Th = params['Th'].astype(np.float32)  # Global translation vector
    R = cv2.Rodrigues(Rh)[0].astype(np.float32)  # Convert rotation vector to matrix

    # Transform vertices to SMPL coordinate system
    smpl_vertices = np.dot(vertices - Th, R)  # Transform to SMPL space

    # Load SMPL model and faces
    smpl = read_pickle(smpl_model_path)
    faces = smpl['f']

    # Create the mesh in SMPL coordinate system
    smpl_mesh = Mesh(smpl_vertices, faces)

    # Generate a 3D grid in SMPL space
    pts = get_grid_points(smpl_vertices)  # 3D grid around SMPL vertices
    sh = pts.shape  # (d, h, w, 3)
    pts = pts.reshape(-1, 3)  # Flatten grid to (N, 3)

    # Compute closest faces and barycentric coordinates
    closest_face, closest_points = smpl_mesh.closest_faces_and_points(pts)
    vert_ids, bary_coords = smpl_mesh.barycentric_coordinates_for_points(
        closest_points, closest_face.astype('int32'))

    # Perform barycentric interpolation to compute UVs
    uvs = barycentric_interpolation(uv_model['uv'][vert_ids], bary_coords)

    # Reshape UVs to (d, h, w, 2)
    uvs = uvs.reshape(*sh[:3], 2).astype(np.float32)

    # Save UV coordinates to file
    uv_path = os.path.join(output_dir, f'{frame}.npy')
    np.save(uv_path, uvs)
    return uv_path


def generate_uvs_for_all_frames(params_dir, vertices_dir, output_dir, uv_model, smpl_model_path, frame_range):
    """
    Generate UV mapping for all frames in the specified range.
    """
    os.makedirs(output_dir, exist_ok=True)  # Create output directory

    with tqdm(range(frame_range[0], frame_range[1], frame_range[2]), desc="Processing Frames") as pbar:
        for frame in pbar:
            uv_path = get_uv_per_frame(frame, params_dir, vertices_dir, output_dir, uv_model, smpl_model_path)
            pbar.set_postfix_str(f"Saved frame {frame} to {uv_path}")
    log(f"All frames processed. UV files saved to {output_dir}.")



if __name__ == '__main__':
    """
    Generate UV mappings in shape (d, h, w, 2) for all frames in a dataset.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data", type=str)
    parser.add_argument("--output_root", default="data", type=str)
    parser.add_argument("--smpl_model_path", default="SMPL_NEUTRAL.pkl", type=str)
    parser.add_argument("--smpl_uv_path", default="smpl_uv.obj", type=str)
    parser.add_argument('--ranges', type=int, default=None, nargs=3)
    args = parser.parse_args()

    data_root = args.data_root
    output_root = args.output_root
    smpl_model_path = args.smpl_model_path
    smpl_uv_path = args.smpl_uv_path
    frame_range = args.ranges

    # Load UV model
    uv_model = load_obj(smpl_uv_path)

    # Define input and output directories
    params_dir = os.path.join(data_root, 'smpl_params')
    vertices_dir = os.path.join(data_root, 'smpl_vertices')
    uv_output_dir = os.path.join(output_root, 'uv_coordinates')

    # Generate UV mapping for all frames
    generate_uvs_for_all_frames(params_dir, vertices_dir, uv_output_dir, uv_model, smpl_model_path, frame_range)
