import torch.nn as nn
#import spconv
import torch.nn.functional as F
import torch
from lib.config import cfg
from lib.utils.blend_utils import *
from .. import embedder
from lib.utils import net_utils
from typing import *
# from hash_embedder import HashEmbedder
from lib.networks.make_network import make_part_network, make_b_deformer, make_f_deformer
from termcolor import cprint
import pytorch3d.transforms as tfm


def gradient(input: torch.Tensor, output: torch.Tensor, d_out: torch.Tensor = None, create_graph: bool = False, retain_graph: bool = False):
    if d_out is not None:
        d_output = d_out
    elif isinstance(output, torch.Tensor):
        d_output = torch.ones_like(output, requires_grad=False)
    else:
        d_output = [torch.ones_like(o, requires_grad=False) for o in output]
    grads = torch.autograd.grad(inputs=input,
                                outputs=output,
                                grad_outputs=d_output,
                                create_graph=create_graph,
                                retain_graph=retain_graph,
                                only_inputs=True,)
    if len(grads) == 1:
        return grads[0]  # return the gradient directly
    else:
        return grads  # to be expanded


def compute_val_pair_around_range(pts: torch.Tensor, decoder: Callable[[torch.Tensor], torch.Tensor], diff_range: float, precomputed: torch.Tensor = None):
    # sample around input point and compute values
    # pts and its random neighbor are concatenated in second dimension
    # if needed, decoder should return multiple values together to save computation
    n_batch, n_pts, D = pts.shape
    # print_cyan(n_pts)
    neighbor = pts + (torch.rand_like(pts) - 0.5) * diff_range
    if precomputed is None:
        full_pts = torch.cat([pts, neighbor], dim=1)  # cat in n_masked dim
        raw: torch.Tensor = decoder(full_pts)  # (n_batch, n_masked, 3)
    else:
        nei = decoder(neighbor)  # (n_batch, n_masked, 3)
        raw = torch.cat([precomputed, nei], dim=1)
    return raw


class GradModule(nn.Module):
    # GradModule is a module that takes gradient based on whether we're in training mode or not
    # Avoiding the high memory cost of retaining graph of *not needed* back porpagation
    def __init__(self):
        super(GradModule, self).__init__()

    def gradient(self, input: torch.Tensor, output: torch.Tensor, d_out: torch.Tensor = None, create_graph: bool = False, retain_graph: bool = False) -> torch.Tensor:
        return gradient(input, output, d_out, self.training or create_graph, self.training or retain_graph)

    def jacobian(self, input: torch.Tensor, output: torch.Tensor):
        with torch.enable_grad():
            outputs = output.split(1, dim=-1)
        grads = [self.gradient(input, o, retain_graph=(i < len(outputs))) for i, o in enumerate(outputs)]
        jac = torch.stack(grads, dim=-1)
        return jac


class Network(GradModule):
    def __init__(self, init_network=True):
        super(Network, self).__init__()

        if not cfg.part_deform:
            self.tpose_deformer = make_b_deformer(cfg)
            self.fpose_deformer = make_f_deformer(cfg)
        self.tpose_human = TPoseHuman()
        self.albedo_latent = nn.Embedding(6890, 16)

        assert cfg.use_knn

    def pose_points_to_tpose_points(self, pose_pts, pose_dirs, batch):
        """
        pose_pts: n_batch, n_point, 3
        """
        # initial blend weights of points at i
        # save_point_cloud(pose_pts.squeeze()[:64], "debug/pose_pts_{}.ply".format(get_time()))
        B, N, _ = pose_pts.shape
        assert B == 1
        with torch.no_grad():
            assert batch['ppts'].shape[0] == 1
            # init_pbw = pts_knn_blend_weights_multiassign(pose_pts, batch['ppts'], batch['weights'], batch['parts'])  # (B, N, P, 24)
            init_pbw = pts_knn_blend_weights_multiassign_batch(pose_pts, batch['part_pts'][0], batch['part_pbw'][0], batch['lengths2'][0])  # (B, N, P, 24)
            pred_pbw, pnorm = init_pbw[..., :24], init_pbw[..., 24]
            pflag = pnorm < cfg.smpl_thresh  # (B, N, P)

        pose_pts_part_extend = pose_pts[:, :, None, :].expand(B, N, NUM_PARTS, 3).reshape(B, N*NUM_PARTS, 3)
        origin_pose_pts = pose_pts_part_extend
        pose_dirs_part_extend = pose_dirs[:, :, None, :].expand(B, N, NUM_PARTS, 3).reshape(B, N*NUM_PARTS, 3)
        pred_pbw = pred_pbw.reshape(B, N*NUM_PARTS, 24).permute(0, 2, 1)
        pflag = pflag.reshape(B, N*NUM_PARTS)

        # transform points from i to i_0
        A_bw, R_inv = get_inverse_blend_params(pred_pbw, batch['A'])
        big_A_bw = get_blend_params(pred_pbw, batch['big_A'])

        init_tpose = pose_points_to_tpose_points(pose_pts_part_extend, A_bw=A_bw, R_inv=R_inv)  # (B, N*P, 3)
        init_bigpose = tpose_points_to_pose_points(init_tpose, A_bw=big_A_bw)  # (B, N*P, 3)

        if cfg.tpose_viewdir and pose_dirs is not None:
            init_tdirs = pose_dirs_to_tpose_dirs(pose_dirs_part_extend, A_bw=A_bw, R_inv=R_inv)
            tpose_dirs = tpose_dirs_to_pose_dirs(init_tdirs, A_bw=big_A_bw).reshape(B, N, NUM_PARTS, 3)
        else:
            tpose_dirs = None

        assert cfg.part_deform == False

        resd = self.tpose_deformer(init_bigpose, batch, flag=pflag)
        tpose = init_bigpose + resd

        tpose = tpose.reshape(B, N, NUM_PARTS, 3)
        pflag = pflag.reshape(B, N, NUM_PARTS)
        resd = resd.reshape(B, N, NUM_PARTS, 3)
        init_bigpose = init_bigpose.reshape(B, N, NUM_PARTS, 3)

        # save_point_cloud(tpose.squeeze()[:64], "debug/tpose_pts_{}.ply".format(get_time()))
        return tpose, tpose_dirs, resd, pflag, init_bigpose, pnorm, origin_pose_pts, A_bw, big_A_bw


    def inverse_tpose_points_to_pose_points(self, tpose_pts, A_bw, big_A_bw, pflag, batch):
        """
        pose_pts: n_batch, n_point, 3
        """
        B, N, _, _ = tpose_pts.shape

        pose_pts_part_extend = tpose_pts.reshape(B, N * NUM_PARTS, 3)
        init_tpose = inverse_pose_points_to_tpose_points(pose_pts_part_extend, A_bw=big_A_bw)  # (B, N*P, 3)
        pose = inverse_tpose_points_to_pose_points(init_tpose, A_bw=A_bw)  # (B, N*P, 3)
        pflag = pflag.reshape(B, N*NUM_PARTS)
        resd = self.fpose_deformer(pose, batch, flag=pflag)
        pose = pose + resd

        return pose


    def resd(self, tpts: torch.Tensor, batch):
        B, N, D = tpts.shape
        return self.tpose_deformer(tpts, batch).view(B, N, D)  # has batch dimension

    def forward(self, wpts: torch.Tensor, viewdir: torch.Tensor, dists: torch.Tensor, batch):
        # transform points from the world space to the pose space
        wpts = wpts[None]  # B, N, 3, fake batch dimension
        viewdir = viewdir[None]  # B, N, 3
        pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])
        pose_dirs = world_dirs_to_pose_dirs(viewdir, batch['R'])

        B, N = wpts.shape[:2]
        assert B == 1

        with torch.no_grad():
            pnorm = pts_sample_blend_weights(pose_pts, batch['pbw'][..., -1:], batch['pbounds']) 
            pnorm = pnorm[0, -1]  # B, 25, N -> N,
            pind = pnorm < cfg.smpl_thresh  # N,
            pind = pind.nonzero(as_tuple=True)[0][:, None].expand(-1, 3)  # N, remove uncessary sync
            viewdir = viewdir[0].gather(dim=0, index=pind)[None]
            pose_pts = pose_pts[0].gather(dim=0, index=pind)[None]
            pose_dirs = pose_dirs[0].gather(dim=0, index=pind)[None]

        # transform points from the pose space to the tpose space
        tpose, tpose_dirs, resd, tpose_part_flag, tpts, part_dist, origin_pose_pts, A_bw, big_A_bw = self.pose_points_to_tpose_points(pose_pts, pose_dirs, batch)
        pose = self.inverse_tpose_points_to_pose_points(tpose, A_bw, big_A_bw, tpose_part_flag, batch)

        # consis = origin_pose_pts[tpose_part_flag] - pose[tpose_part_flag]
        tpose_part_flag = tpose_part_flag[0]
        plag = tpose_part_flag.view(-1)
        c_pts = origin_pose_pts[0][plag]
        c_pose = pose[0][plag]
        distance = torch.norm(c_pts - c_pose, dim=1)
        loss_consis = distance[distance > 0.05]

        tpose = tpose[0]

        if cfg.tpose_viewdir:
            viewdir = tpose_dirs[0]
        else:
            viewdir = viewdir[0]

        # part network query with hashtable embedding
        ret = self.tpose_human(tpose, viewdir, tpose_part_flag, dists, part_dist, batch)


        raw = torch.zeros((B, N, 4), device=wpts.device, dtype=wpts.dtype)
        occ = torch.zeros((B, N, 1), device=wpts.device, dtype=wpts.dtype)
        raw[0, pind[:, 0]] = ret['raw']  # NOTE: ignoring batch dimension
        occ[0, pind[:, 0]] = ret['occ']

        ret.update({'raw': raw, 'occ': occ, 'loss_consis': loss_consis})
        # ret.update({'raw': raw, 'occ': occ})

        if self.training:
            tocc = ret['tocc'].view(B, -1, 1)  # tpose occ, correpond with tpts (before deform)
            ret.update({'resd': resd, 'tpts': tpts, 'tocc': tocc})
        else:
            del ret['tocc']
        return ret


class TPoseHuman(GradModule):
    def __init__(self):
        super(TPoseHuman, self).__init__()

        # self.head_bbox = torch.Tensor([ # 313
        #     [-0.3, 0.1, -0.3],
        #     [0.3, 0.7, 0.3]
        # ]).cuda()

        self.part_networks = nn.ModuleList([make_part_network(cfg, p, i) for i, p in enumerate(partnames)])
        self.vec_mode = [0, 1, 2]
        self.pose_dim = 20
        self.n_comp = [48, 48, 48]
        # self.grid_size = [512, 512, 128]
        self.grid_size = [512, 512, 512]
        self.pose_num = cfg.key_pose_num
        self.init_svd_volume()
        self.mat_mode = [[0, 1], [0, 2], [1, 2]]
        if not cfg.silent:
            print("Finish initialize part networks")

    def save_part_decoders(self):
        for partnet in self.part_networks:
            partnet.save_decoder()
        self.body_network.save_decoder()

    def save_parts(self):
        for partnet in self.part_networks:
            partnet.save_part()
        self.body_network.save_part()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_part_pose(self, pose, part_indices):
        part_pose = []
        for idx in part_indices:
            start_idx = idx * 3
            part_pose.append(pose[..., start_idx:start_idx + 3])
        if isinstance(pose, np.ndarray):
            part_pose = [torch.from_numpy(p) for p in part_pose]
        part_pose = torch.stack(part_pose, dim=-2)
        return part_pose

    def get_pose(self, pose, part_indices):
        part_pose = []
        for idx in part_indices:
            start_idx = idx * 3
            part_pose.append(pose[..., start_idx:start_idx + 3])
        if isinstance(pose, np.ndarray):
            part_pose = [torch.from_numpy(p) for p in part_pose]
        part_pose = torch.stack(part_pose, dim=-2)
        part_pose = part_pose.reshape(*part_pose.shape[:-2], -1)
        return part_pose

    def get_pose_feat(self, pts, batch, part_idx):

        k = 1

        pose = batch['poses']

        key_pose = batch['all_poses']

        # key_quat = tfm.axis_angle_to_quaternion(key_pose.reshape((*key_pose.shape[:2], 23, 3)))
        # query_quat = tfm.axis_angle_to_quaternion(pose.reshape(1, 23, 3))
        # pose_dist = torch.abs((query_quat[:, None, :] * key_quat).sum(-1)).sum(-1)

        part_indices = part_bw_map[partnames[part_idx]]
        key_part_pose = self.get_part_pose(key_pose, part_indices)
        key_part_quat = tfm.axis_angle_to_quaternion(key_part_pose)
        query_part_pose = self.get_part_pose(pose, part_indices)
        query_part_quat = tfm.axis_angle_to_quaternion(query_part_pose)
        pose_dist = torch.abs((query_part_quat[:, None, :] * key_part_quat).sum(-1)).sum(-1)

        topk_weight, topk_id = torch.topk(pose_dist, k, dim=-1)  # (B, J, K)

        topk_weight = F.normalize(topk_weight, dim=-1, p=1, eps=1e-16)
        pts = self.normalize_pts(pts, batch['tbounds'][0])

        pose_feats = self.get_svd_feat(pts, topk_id, k)
        pose_feat = torch.sum(topk_weight[0, :, None, None] * pose_feats, dim=0, keepdim=True)
        # if cfg.concat_pose:
        #     # pose = pose[:, None].repeat(1, pts.shape[1], 1)
        #     pose = self.get_pose(pose, part_indices).repeat(1, pts.shape[1], 1)
        #     pose_feat = torch.cat([pose_feat, pose], dim=-1)

        return pose_feat

    def normalize_pts(self, pts, bounds):
        pts = pts.detach()
        xyz_min = bounds[0]
        xyz_max = bounds[1]
        normalized_pts = (pts - xyz_min) / (xyz_max - xyz_min) * 2 - 1
        return normalized_pts

    def get_svd_feat(self, xyz_sampled, idx, k):

        coordinate_line = torch.stack(
            (xyz_sampled[..., self.vec_mode[0]],
             xyz_sampled[..., self.vec_mode[1]],
             xyz_sampled[..., self.vec_mode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).view(3, -1, 1, 2)

        coord = []
        feat = []

        for idx_line in range(len(self.coord_line)):
            # The current F.grif_sample in pytorch doesn't support backpropagation

            line_coef_point = net_utils.grid_sample(self.coord_line[idx_line][idx[0]],coordinate_line[[idx_line]].repeat(k, 1, 1, 1)).view(k, -1, *xyz_sampled.shape[:2])  # [16, h*w*n]
            feat_coef = self.feat_line[idx_line][idx[0]].repeat(1, 1, 1, xyz_sampled.shape[1])
            coord.append(line_coef_point)
            feat.append(feat_coef)
        pose_features = []

        for idx_plane in range(len(self.mat_mode)):
            id0, id1 = self.mat_mode[idx_plane]

            plane = torch.sum(coord[id0] * coord[id1] * feat[idx_plane], dim=1)
            pose_features.append(plane)

        pose_feature = torch.cat(pose_features, dim=1)
        return pose_feature.permute(0, 2, 1)

    def init_svd_volume(self):
        self.coord_line, self.feat_line = self.init_one_svd(self.n_comp, self.grid_size, 0.1)

    def init_one_svd(self, n_component, gridSize, scale):
        line_coef = []
        feat_coef = []
        for i in range(len(self.vec_mode)):
            vec_id = self.vec_mode[i]
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((self.pose_num, n_component[i], gridSize[vec_id], 1))))
            feat_coef.append(
                torch.nn.Parameter(scale * torch.randn((self.pose_num, n_component[i], self.pose_dim, 1)))
            )
        return torch.nn.ParameterList(line_coef), torch.nn.ParameterList(feat_coef)

    def forward(self, tpts: torch.Tensor, viewdir: torch.Tensor, tflag: torch.Tensor, dists: torch.Tensor, part_dist, batch):
        """
        """
        assert not cfg.part_deform

        # prepare inputs
        N = tpts.shape[0]
        raws = torch.zeros(N, NUM_PARTS, 4, device=tpts.device)
        occs = torch.zeros(N, NUM_PARTS, 1, device=tpts.device)

        # computing indices
        inds = []
        for part_idx in range(NUM_PARTS):
            flag_part = tflag[:, part_idx]  # flag_part: N, assuming viewdir and xyz have same dim
            flag_inds = flag_part.nonzero(as_tuple=True)[0]  # When input is on CUDA, torch.nonzero() causes host-device synchronization.
            inds.append(flag_inds)

        # applying mask
        xyz_parts = []
        viewdir_parts = []
        for part_idx in range(NUM_PARTS):
            xyz_part = tpts[:, part_idx].gather(dim=0, index=inds[part_idx][:, None].expand(-1, 3))  # faster backward than indexing, using resd so need backward
            viewdir_part = viewdir[:, part_idx].gather(dim=0, index=inds[part_idx][:, None].expand(-1, 3))
            xyz_parts.append(xyz_part)
            viewdir_parts.append(viewdir_part)

        # forward network
        ret_parts = []
        for part_idx in range(NUM_PARTS):
            xyz_part = xyz_parts[part_idx]
            viewdir_part = viewdir_parts[part_idx]
            if xyz_part is None or xyz_part.numel() == 0:
                # feat = torch.zeros((1, xyz_part.shape[0], getattr(cfg.partnet, partnames[part_idx]).dim), device='cuda')
                feat = torch.zeros((1, xyz_part.shape[0], 60), device='cuda')
            else:
                feat = self.get_pose_feat(xyz_part[None], batch, part_idx)
            part_network = self.part_networks[part_idx]
            ret_part = part_network(xyz_part, viewdir_part, dists, batch, feat[0])
            # ret_part = part_network(xyz_part, viewdir_part, dists, batch)
            ret_parts.append(ret_part)


        # fill in output
        for part_idx in range(NUM_PARTS):
            flag_inds = inds[part_idx]
            ret_part = ret_parts[part_idx]
            raws[flag_inds, part_idx] = ret_part['raw'].to(raws.dtype, non_blocking=True)
            occs[flag_inds, part_idx] = ret_part['occ'].to(occs.dtype, non_blocking=True)

        if cfg.aggr == 'mean':
            raw = raws.mean(dim=1)
            occ = occs.mean(dim=1)
            return {'raw': raw, 'occ': occ, 'tocc': occs}
        elif cfg.aggr == 'dist':
            part_dist_inv = F.normalize(1.0 / (part_dist[0] + 1e-5), dim=-1)
            raw = torch.sum(raws * part_dist_inv[..., None], dim=1)
            occ = torch.sum(occs * part_dist_inv[..., None], dim=1)
            return {'raw': raw, 'occ': occ, 'tocc': occs}
        elif cfg.aggr == 'mindist':
            breakpoint()
            ind = part_dist[0, :, :, None].argmin(dim=1).reshape(N, 1, 1).expand(N, 1, 4)
            raw = torch.gather(raws, 1, ind)[:, 0, :]
            ind = part_dist[0, :, :, None].argmin(dim=1).reshape(N, 1, 1).expand(N, 1, 1)
            occ = torch.gather(occs, 1, ind)[:, 0, :]
            return {'raw': raw, 'occ': occ, 'tocc': occs}

        ind = occs.argmax(dim=1).reshape(N, 1, 1).expand(N, 1, 4)
        raw = torch.gather(raws, 1, ind)[:, 0, :]
        occ = occs.max(dim=1)[0]
        return {'raw': raw, 'occ': occ, 'tocc': occs}

    def get_occupancy(self, tpts, batch):
        raise NotImplementedError
