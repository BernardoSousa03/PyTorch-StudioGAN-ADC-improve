# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/metrics/preparation.py

from os.path import exists, join
import os
import pickle

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import torch.nn.modules.linear as nn_linear

# Torchvision compatibility shims for older pickled transforms.
if not hasattr(transforms.Resize, "max_size"):
    transforms.Resize.max_size = None
if not hasattr(transforms.Resize, "antialias"):
    transforms.Resize.antialias = None
if not hasattr(nn.MultiheadAttention, "batch_first"):
    nn.MultiheadAttention.batch_first = False

from metrics.inception_net import InceptionV3
from metrics.swin_transformer import SwinTransformer
import metrics.features as features
import metrics.vit as vits
import metrics.fid as fid
import metrics.ins as ins
import utils.misc as misc
import utils.ops as ops
import utils.resize as resize

model_versions = {"InceptionV3_torch": "pytorch/vision:v0.10.0",
                  "ResNet_torch": "pytorch/vision:v0.10.0",
                  "SwAV_torch": "facebookresearch/swav:main"}
VGG16_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2_ada/versions/1/files/vgg16.pt"  # reference only
model_names = {"InceptionV3_torch": "inception_v3",
               "ResNet50_torch": "resnet50",
               "SwAV_torch": "resnet50"}
SWAV_CLASSIFIER_URL = "https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_eval_linear.pth.tar"
SWIN_URL = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth"


class LoadEvalModel(object):
    def __init__(self, eval_backbone, post_resizer, world_size, distributed_data_parallel, device,
                 clip_pkl_path=None, vgg16_pkl_path=None):
        super(LoadEvalModel, self).__init__()
        self.eval_backbone = eval_backbone
        self.post_resizer = post_resizer
        self.device = device
        self.save_output = misc.SaveOutput()

        if self.eval_backbone == "InceptionV3_tf":
            self.res, mean, std = 299, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            self.model = InceptionV3(resize_input=False,
                                     normalize_input=False).to(self.device)
        elif self.eval_backbone in ["InceptionV3_torch", "ResNet50_torch", "SwAV_torch"]:
            self.res = 299 if "InceptionV3" in self.eval_backbone else 224
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            self.model = torch.hub.load(model_versions[self.eval_backbone],
                                        model_names[self.eval_backbone],
                                        pretrained=True)
            if self.eval_backbone == "SwAV_torch":
                linear_state_dict = load_state_dict_from_url(SWAV_CLASSIFIER_URL, progress=True)["state_dict"]
                linear_state_dict = {k.replace("module.linear.", ""): v for k, v in linear_state_dict.items()}
                self.model.fc.load_state_dict(linear_state_dict, strict=True)
            self.model = self.model.to(self.device)
            hook_handles = []
            for name, layer in self.model.named_children():
                if name == "fc":
                    handle = layer.register_forward_pre_hook(self.save_output)
                    hook_handles.append(handle)
        elif self.eval_backbone == "DINO_torch":
            self.res, mean, std = 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            self.model = vits.__dict__["vit_small"](patch_size=8, num_classes=1000, num_last_blocks=4)
            misc.load_pretrained_weights(self.model, "", "teacher", "vit_small", 8)
            misc.load_pretrained_linear_weights(self.model.linear, "vit_small", 8)
            self.model = self.model.to(self.device)
        elif self.eval_backbone == "Swin-T_torch":
            self.res, mean, std = 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            self.model = SwinTransformer()
            model_state_dict = load_state_dict_from_url(SWIN_URL, progress=True)["model"]
            self.model.load_state_dict(model_state_dict, strict=True)
            self.model = self.model.to(self.device)
        elif self.eval_backbone == "CLIP_torch":
            # Loads a self-contained CLIP visual encoder pkl (no 'clip' package required).
            # The pkl model handles its own resizing and normalisation internally;
            # it expects uint8 images in [0, 255] and returns L2-normalised embeddings.
            # Compatible with the clip-vit_b32.pkl format from Unconditional-Training-CGANs.
            assert clip_pkl_path is not None, \
                "Please supply --clip_pkl_path pointing to clip-vit_b32.pkl (or equivalent)."
            assert exists(clip_pkl_path), \
                f"CLIP pkl not found at: {clip_pkl_path}"
            if not hasattr(nn_linear, "_LinearWithBias"):
                # Unpickle compatibility for older CLIP pickles.
                nn_linear._LinearWithBias = nn.Linear
            self.res, mean, std = 224, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]  # unused; model preprocesses internally
            with open(clip_pkl_path, "rb") as f:
                self.model = pickle.load(f).eval().to(self.device)
            self._patch_torchvision_transforms()
        elif self.eval_backbone == "VGG16_torch":
            # Loads a self-contained VGG16 TorchScript model (no extra package required).
            # The model handles its own resizing and normalisation internally;
            # it expects uint8 images in [0, 255]. Call with return_features=True to get
            # feature vectors (2048-d) instead of class logits.
            # Compatible with the vgg16.pt format from Unconditional-Training-CGANs /
            # StyleGAN2-ADA (https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2_ada).
            assert vgg16_pkl_path is not None, \
                "Please supply --vgg16_pkl_path pointing to vgg16.pt (or equivalent)."
            assert exists(vgg16_pkl_path), \
                f"VGG16 model not found at: {vgg16_pkl_path}"
            self.res, mean, std = 224, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]  # unused; model preprocesses internally
            with open(vgg16_pkl_path, "rb") as f:
                self.model = torch.jit.load(f).eval().to(self.device)
        else:
            raise NotImplementedError

        self.resizer = resize.build_resizer(resizer=self.post_resizer, backbone=self.eval_backbone, size=self.res)
        self.totensor = transforms.ToTensor()
        self.mean = torch.Tensor(mean).view(1, 3, 1, 1).to(self.device)
        self.std = torch.Tensor(std).view(1, 3, 1, 1).to(self.device)

        if world_size > 1 and distributed_data_parallel:
            misc.make_model_require_grad(self.model)
            self.model = DDP(self.model,
                             device_ids=[self.device],
                             broadcast_buffers=False if self.eval_backbone=="Swin-T_torch" else True)
        elif world_size > 1 and distributed_data_parallel is False:
            self.model = DataParallel(self.model, output_device=self.device)
        else:
            pass

    def eval(self):
        self.model.eval()

    def _patch_torchvision_transforms(self):
        # Some pickles store torchvision transform objects without newer attributes.
        try:
            default_interp = transforms.InterpolationMode.BILINEAR
        except AttributeError:
            default_interp = 2  # BILINEAR

        for module in self.model.modules():
            if isinstance(module, transforms.Resize):
                if not hasattr(module, "max_size"):
                    module.max_size = None
                if not hasattr(module, "antialias"):
                    module.antialias = None
                if not hasattr(module, "interpolation"):
                    module.interpolation = default_interp
            elif isinstance(module, transforms.RandomResizedCrop):
                if not hasattr(module, "antialias"):
                    module.antialias = None
                if not hasattr(module, "interpolation"):
                    module.interpolation = default_interp
            elif isinstance(module, (transforms.CenterCrop, transforms.RandomCrop)):
                if not hasattr(module, "interpolation"):
                    module.interpolation = default_interp

    def get_outputs(self, x, quantize=False):
        if self.eval_backbone in ("CLIP_torch", "VGG16_torch"):
            # These self-contained pkl/pt models handle their own preprocessing;
            # feed raw uint8 images directly (no external normalisation).
            if quantize:
                x_uint8 = ops.quantize_images(x)           # returns uint8 numpy array
                if isinstance(x_uint8, np.ndarray):
                    x_uint8 = torch.from_numpy(x_uint8)
            else:
                x_uint8 = x.clamp(0, 255).to(torch.uint8)  # already on GPU
            x_uint8 = x_uint8.to(self.device)
            with torch.no_grad():
                if self.eval_backbone == "CLIP_torch":
                    repres = self.model(x_uint8)
                else:  # VGG16_torch — call with return_features=True to get feature vectors
                    repres = self.model(x_uint8, return_features=True)
            # Return a dummy logits tensor so callers expecting (repres, logits) work.
            # IS is meaningless with CLIP/VGG16; FID/PRDC only consume repres.
            dummy_logits = torch.zeros(repres.shape[0], 1000, device=self.device)
            return repres, dummy_logits

        if quantize:
            x = ops.quantize_images(x)
        else:
            x = x.detach().cpu().numpy().astype(np.uint8)
        x = ops.resize_images(x, self.resizer, self.totensor, self.mean, self.std, device=self.device)

        if self.eval_backbone in ["InceptionV3_tf", "DINO_torch", "Swin-T_torch"]:
            repres, logits = self.model(x)
        elif self.eval_backbone in ["InceptionV3_torch", "ResNet50_torch", "SwAV_torch"]:
            logits = self.model(x)
            if len(self.save_output.outputs) > 1:
                repres = []
                for rank in range(len(self.save_output.outputs)):
                    repres.append(self.save_output.outputs[rank][0].detach().cpu())
                repres = torch.cat(repres, dim=0).to(self.device)
            else:
                repres = self.save_output.outputs[0][0].to(self.device)
            self.save_output.clear()
        return repres, logits


def prepare_moments(data_loader, eval_model, quantize, cfgs, logger, device):
    disable_tqdm = device != 0
    eval_model.eval()
    moment_dir = join(cfgs.RUN.save_dir, "moments")
    if not exists(moment_dir):
        os.makedirs(moment_dir)
    moment_path = join(moment_dir, cfgs.DATA.name + "_"  + str(cfgs.DATA.img_size) + "_"+ cfgs.RUN.pre_resizer + "_" + \
                       cfgs.RUN.ref_dataset + "_" + cfgs.RUN.post_resizer + "_" + eval_model.eval_backbone + "_moments.npz")

    is_file = os.path.isfile(moment_path)
    if is_file:
        mu = np.load(moment_path)["mu"]
        sigma = np.load(moment_path)["sigma"]
    else:
        if device == 0:
            logger.info("Calculate moments of {ref} dataset using {eval_backbone} model.".\
                        format(ref=cfgs.RUN.ref_dataset, eval_backbone=cfgs.RUN.eval_backbone))
        mu, sigma = fid.calculate_moments(data_loader=data_loader,
                                          eval_model=eval_model,
                                          num_generate="N/A",
                                          batch_size=cfgs.OPTIMIZATION.batch_size,
                                          quantize=quantize,
                                          world_size=cfgs.OPTIMIZATION.world_size,
                                          DDP=cfgs.RUN.distributed_data_parallel,
                                          disable_tqdm=disable_tqdm,
                                          fake_feats=None)

        if device == 0:
            logger.info("Save calculated means and covariances to disk.")
        np.savez(moment_path, **{"mu": mu, "sigma": sigma})
    return mu, sigma


def prepare_real_feats(data_loader, eval_model, num_feats, quantize, cfgs, logger, device):
    disable_tqdm = device != 0
    eval_model.eval()
    feat_dir = join(cfgs.RUN.save_dir, "feats")
    if not exists(feat_dir):
        os.makedirs(feat_dir)
    feat_path = join(feat_dir, cfgs.DATA.name + "_"  + str(cfgs.DATA.img_size) + "_"+ cfgs.RUN.pre_resizer + "_" + \
                     cfgs.RUN.ref_dataset + "_" + cfgs.RUN.post_resizer + "_" + eval_model.eval_backbone + "_feats.npz")

    is_file = os.path.isfile(feat_path)
    if is_file:
        real_feats = np.load(feat_path)["real_feats"]
    else:
        if device == 0:
            logger.info("Calculate features of {ref} dataset using {eval_backbone} model.".\
                        format(ref=cfgs.RUN.ref_dataset, eval_backbone=eval_model.eval_backbone))
        real_feats, real_probs, real_labels = features.stack_features(data_loader=data_loader,
                                                eval_model=eval_model,
                                                num_feats=num_feats,
                                                batch_size=cfgs.OPTIMIZATION.batch_size,
                                                quantize=quantize,
                                                world_size=cfgs.OPTIMIZATION.world_size,
                                                DDP=cfgs.RUN.distributed_data_parallel,
                                                device=device,
                                                disable_tqdm=disable_tqdm)
        if device == 0:
            logger.info("Save real_features to disk.")
            np.savez(feat_path, **{"real_feats": real_feats,
                                   "real_probs": real_probs,
                                   "real_labels": real_labels})
    return real_feats


def calculate_ins(data_loader, eval_model, quantize, splits, cfgs, logger, device):
    disable_tqdm = device != 0
    is_acc = True if "ImageNet" in cfgs.DATA.name and "Tiny" not in cfgs.DATA.name else False
    if device == 0:
        logger.info("Calculate inception score of the {ref} dataset uisng pre-trained {eval_backbone} model.".\
                    format(ref=cfgs.RUN.ref_dataset, eval_backbone=eval_model.eval_backbone))
    is_score, is_std, top1, top5 = ins.eval_dataset(data_loader=data_loader,
                                                    eval_model=eval_model,
                                                    quantize=quantize,
                                                    splits=splits,
                                                    batch_size=cfgs.OPTIMIZATION.batch_size,
                                                    world_size=cfgs.OPTIMIZATION.world_size,
                                                    DDP=cfgs.RUN.distributed_data_parallel,
                                                    is_acc=is_acc,
                                                    is_torch_backbone=True if "torch" in eval_model.eval_backbone else False,
                                                    disable_tqdm=disable_tqdm)
    if device == 0:
        logger.info("Inception score={is_score}-Inception_std={is_std}".format(is_score=is_score, is_std=is_std))
        if is_acc:
            logger.info("{eval_model} Top1 acc: ({num} images): {Top1}".format(
                eval_model=eval_model.eval_backbone, num=str(len(data_loader.dataset)), Top1=top1))
            logger.info("{eval_model} Top5 acc: ({num} images): {Top5}".format(
                eval_model=eval_model.eval_backbone, num=str(len(data_loader.dataset)), Top5=top5))
