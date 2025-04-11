import torch
import torch.nn.functional as F
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
from lib.utils.blend_utils import *


class Renderer:
    def __init__(self, net):
        self.net = net

    def render(self, batch, test=False, epoch=-1):
        pts = batch['pts']
        sh = pts.shape

        if epoch != -1:
            batch['epoch'] = epoch

        if 'latent_index' not in batch:
            batch['latent_index'] = 0

        # volume rendering for each pixel
        chunk = 4096 * 32
        tpts = pts.reshape(-1, 3)
        N = tpts.shape[0]
        ret_list = []
        # print(ray_o.shape)
        # print(batch['mask_at_box'].shape)
        from tqdm import tqdm
        for i in tqdm(range(0, N, chunk)):
            pts = tpts[i:i + chunk]
            viewdir = torch.zeros_like(pts)
            ret = self.net(pts, viewdir, None, batch)
            ret_list.append({
                "occ": ret['occ'][0]
            })

        breakpoint()

        keys = ret_list[0].keys()
        ret = {k: torch.cat([r[k] for r in ret_list], dim=0) for k in keys}
        assert "occ" in ret.keys() and len(ret.keys()) == 1

        ret['occ'] = ret['occ'].view(sh[1:-1]).detach().cpu().numpy()

        return ret