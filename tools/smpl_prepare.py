import os
import numpy as np
import smplx  # 确保你有安装 SMPL 模型库

# 设置输入和输出路径
input_path = "/data/ljw/test/monocap/lan_images620_1300"
output_path = os.path.join(input_path, "output")  # 生成 smpl_params, smpl_lbs, smpl_vertices 的路径
os.makedirs(output_path, exist_ok=True)

# 加载 SMPL 模型
model_path = "/data/ljw/test/pb-nerf/data/smpl-meta/SMPL_NEUTRAL.pkl"  # 替换为实际的模型路径
smpl_model = smplx.create(model_path, model_type='smpl')

# 定义文件夹路径
params_path = os.path.join(input_path, "params")
lbs_path = os.path.join(input_path, "lbs")
vertices_path = os.path.join(input_path, "vertices")

# 定义输出文件夹
smpl_params_path = os.path.join(output_path, "smpl_params")
smpl_lbs_path = os.path.join(output_path, "smpl_lbs")
smpl_vertices_path = os.path.join(output_path, "smpl_vertices")
os.makedirs(smpl_params_path, exist_ok=True)
os.makedirs(smpl_lbs_path, exist_ok=True)
os.makedirs(smpl_vertices_path, exist_ok=True)

# 遍历 params 文件夹中的文件
for file_name in os.listdir(params_path):
    if file_name.endswith(".npy"):  # 假设文件为 npy 格式
        # 读取 params 文件
        params = np.load(os.path.join(params_path, file_name), allow_pickle=True).item()

        # 生成 smpl_params
        smpl_params = {
            'pose': params['pose'],
            'shape': params['shape'],
            'trans': params['trans']
        }
        # 保存 smpl_params 为 npy 格式
        np.save(os.path.join(smpl_params_path, file_name), smpl_params)

        # 通过 SMPL 模型生成 smpl_lbs 和 smpl_vertices
        smpl_output = smpl_model(
            global_orient=smpl_params['pose'][:, :3],
            body_pose=smpl_params['pose'][:, 3:],
            betas=smpl_params['shape'],
            transl=smpl_params['trans']
        )

        # 提取顶点信息
        smpl_vertices = smpl_output.vertices.detach().cpu().numpy()

        # 保存 smpl_lbs 和 smpl_vertices 为 npy 格式
        np.save(os.path.join(smpl_lbs_path, file_name), smpl_output)
        np.save(os.path.join(smpl_vertices_path, file_name), smpl_vertices)

print("All files have been processed and saved in the output folder.")
