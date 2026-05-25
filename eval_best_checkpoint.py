#!/usr/bin/env python3
"""
Evaluate metrics (FID, FID-specific classes, PRDC) on the best checkpoint.

Usage:
    python eval_best_checkpoint.py --cfg src/configs/FlowersLT/BigGAN-ADC-DiffAug.yaml \\
                                    --ckpt_path results/checkpoint_best.pth
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_util import Dataset_
from config import Configurations
from models import model as model_module
import metrics.preparation as pp
import metrics.features as features
import metrics.fid as fid
import metrics.prdc as prdc
import utils.misc as misc


def load_checkpoint(ckpt_path, device='cuda'):
    """Load checkpoint and return state dict."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt


def build_generator(cfgs, device='cuda'):
    """Build and return the generator model."""
    Gen = model_module.Generator(
        z_dim=cfgs.MODEL.z_dim,
        shared_dim=cfgs.MODEL.g_shared_dim,
        img_size=cfgs.DATA.img_size,
        g_conv_dim=cfgs.MODEL.g_conv_dim,
        g_spectral_norm=cfgs.MODEL.apply_g_sn,
        attention=cfgs.MODEL.apply_attn,
        attention_after_layer=cfgs.MODEL.attn_g_loc,
        activation_fn=cfgs.MODEL.activation_fn,
        conditional_strategy=cfgs.MODEL.g_cond_mtd,
        num_classes=cfgs.DATA.num_classes,
        g_init=cfgs.MODEL.g_init,
        g_depth=cfgs.MODEL.g_depth,
        mixed_precision=cfgs.RUN.mixed_precision
    ).to(device)
    
    return Gen


def evaluate_metrics(cfgs, ckpt_path, device='cuda:0'):
    """
    Evaluate FID, FID-specific classes, and PRDC metrics on the best checkpoint.
    
    Args:
        cfgs: Configuration object
        ckpt_path: Path to the checkpoint file
        device: Device to run evaluation on
    """
    
    # Set random seeds
    misc.fix_seed(cfgs.RUN.seed)
    
    # Load evaluation dataset
    print(f"Loading {cfgs.DATA.name} {cfgs.RUN.ref_dataset} dataset for evaluation...")
    eval_dataset = Dataset_(
        data_name=cfgs.DATA.name,
        data_dir=cfgs.RUN.data_dir,
        train=True if cfgs.RUN.ref_dataset == "train" else False,
        crop_long_edge=False if cfgs.DATA.name in cfgs.MISC.no_proc_data else True,
        resize_size=None if cfgs.DATA.name in cfgs.MISC.no_proc_data else cfgs.DATA.img_size,
        resizer=cfgs.RUN.pre_resizer,
        random_flip=False,
        hdf5_path=None,
        normalize=True,
        load_data_in_memory=False
    )
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    # Create dataloaders
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=cfgs.OPTIMIZATION.batch_size,
        shuffle=False,
        pin_memory=cfgs.RUN.pin_memory,
        num_workers=cfgs.RUN.num_workers,
        drop_last=False
    )
    
    # Load evaluation model (InceptionV3 or ResNet50)
    print(f"Loading evaluation model: {cfgs.RUN.eval_backbone}")
    eval_model = pp.LoadEvalModel(
        eval_backbone=cfgs.RUN.eval_backbone,
        post_resizer=cfgs.RUN.post_resizer,
        world_size=1,
        distributed_data_parallel=False,
        device=device
    )
    
    # Load real image features
    print("Computing real image features...")
    mu_real, sigma_real = pp.prepare_moments(
        data_loader=eval_dataloader,
        eval_model=eval_model,
        quantize=True,
        cfgs=cfgs,
        logger=None,
        device=device
    )
    print(f"Real features shape: mu={mu_real.shape}, sigma={sigma_real.shape}")
    
    # Build generator
    print(f"Building generator from checkpoint: {ckpt_path}")
    Gen = build_generator(cfgs, device=device)
    
    # Load checkpoint weights
    ckpt = load_checkpoint(ckpt_path, device=device)
    if 'state_dict' in ckpt:
        Gen.load_state_dict(ckpt['state_dict'])
    else:
        Gen.load_state_dict(ckpt)
    print("Generator weights loaded successfully")
    
    # Set to evaluation mode
    Gen.eval()
    
    # Prepare discriminator (needed for some model configurations)
    try:
        Dis = model_module.Discriminator(
            img_size=cfgs.DATA.img_size,
            d_conv_dim=cfgs.MODEL.d_conv_dim,
            d_spectral_norm=cfgs.MODEL.apply_d_sn,
            attention=cfgs.MODEL.apply_attn,
            attention_after_layer=cfgs.MODEL.attn_d_loc,
            activation_fn=cfgs.MODEL.activation_fn,
            conditional_strategy=cfgs.MODEL.d_cond_mtd,
            num_classes=cfgs.DATA.num_classes,
            d_init=cfgs.MODEL.d_init,
            d_depth=cfgs.MODEL.d_depth,
            mixed_precision=cfgs.RUN.mixed_precision
        ).to(device)
    except Exception as e:
        print(f"Warning: Could not build discriminator: {e}")
        Dis = None
    
    # Generate fake images and extract features
    print(f"Generating {len(eval_dataset)} fake images and extracting features...")
    num_generate = len(eval_dataset)
    
    fake_feats, fake_probs, fake_labels = features.generate_images_and_stack_features(
        generator=Gen,
        discriminator=Dis,
        eval_model=eval_model,
        num_generate=num_generate,
        y_sampler="totally_random",
        batch_size=cfgs.OPTIMIZATION.batch_size,
        z_prior=cfgs.MODEL.z_prior,
        truncation_factor=cfgs.RUN.truncation_factor,
        z_dim=cfgs.MODEL.z_dim,
        num_classes=cfgs.DATA.num_classes,
        LOSS=cfgs.LOSS,
        RUN=cfgs.RUN,
        MODEL=cfgs.MODEL,
        is_stylegan=False,
        generator_mapping=None,
        generator_synthesis=None,
        quantize=True,
        world_size=1,
        DDP=False,
        device=device,
        logger=None,
        disable_tqdm=False
    )
    print(f"Generated features shape: {fake_feats.shape}")
    
    # Compute FID score
    print("\n" + "="*60)
    print("Computing FID Score...")
    print("="*60)
    fid_score, _, _ = fid.calculate_fid(
        data_loader=eval_dataloader,
        eval_model=eval_model,
        num_generate=num_generate,
        cfgs=cfgs,
        pre_cal_mean=mu_real,
        pre_cal_std=sigma_real,
        fake_feats=fake_feats,
        disable_tqdm=False
    )
    print(f"FID Score: {fid_score:.4f}")
    
    # Compute FID-specific classes
    print("\n" + "="*60)
    print("Computing FID for Specific Classes...")
    print("="*60)
    try:
        fids_by_class = fid.calculate_fid_specific_classes(
            data_loader=eval_dataloader,
            eval_model=eval_model,
            num_generate=num_generate,
            cfgs=cfgs,
            quantize=True,
            fake_feats=fake_feats,
            disable_tqdm=False
        )
        print(f"FID-specific classes computed for {len(fids_by_class)} classes")
        for class_name, class_fid in sorted(fids_by_class.items()):
            print(f"  {class_name}: {class_fid:.4f}")
    except Exception as e:
        print(f"Warning: Could not compute FID-specific classes: {e}")
        fids_by_class = {}
    
    # Compute PRDC
    print("\n" + "="*60)
    print("Computing PRDC (Precision, Recall, Density, Coverage)...")
    print("="*60)
    
    # Prepare real features for PRDC
    real_feats = pp.prepare_feats(
        data_loader=eval_dataloader,
        eval_model=eval_model,
        quantize=True,
        cfgs=cfgs,
        logger=None,
        device=device
    )
    print(f"Real features shape: {real_feats.shape}")
    
    prc, rec, dns, cvg = prdc.calculate_pr_dc(
        real_feats=real_feats,
        fake_feats=fake_feats,
        data_loader=eval_dataloader,
        eval_model=eval_model,
        num_generate=num_generate,
        cfgs=cfgs,
        quantize=True,
        nearest_k=5,
        world_size=1,
        DDP=False,
        disable_tqdm=False
    )
    
    print(f"Improved Precision: {prc:.4f}")
    print(f"Improved Recall: {rec:.4f}")
    print(f"Density: {dns:.4f}")
    print(f"Coverage: {cvg:.4f}")
    
    # Compile results
    results = {
        "checkpoint": ckpt_path,
        "dataset": cfgs.DATA.name,
        "ref_dataset": cfgs.RUN.ref_dataset,
        "num_real_images": len(eval_dataset),
        "num_generated_images": num_generate,
        "metrics": {
            "FID": float(fid_score),
            "FID_specific_classes": fids_by_class,
            "Improved_Precision": float(prc),
            "Improved_Recall": float(rec),
            "Density": float(dns),
            "Coverage": float(cvg)
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate metrics on best checkpoint")
    parser.add_argument('-cfg', '--cfg_file', type=str, required=True, help='Path to config file')
    parser.add_argument('-ckpt', '--ckpt_path', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('-device', '--device', type=str, default='cuda:0', help='Device to run evaluation on')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output JSON file for results (optional)')
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.cfg_file}")
    cfgs = Configurations(args.cfg_file)
    
    # Evaluate metrics
    results = evaluate_metrics(cfgs, args.ckpt_path, device=args.device)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2))
    
    # Save results if output path provided
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
