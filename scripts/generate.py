#!/usr/bin/env python
import os
import glob
import pickle
import argparse

from astropy.io import fits
import numpy as np
import torch
import torch.multiprocessing as mp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# This script must mirror beam/training/trainer.py's model
# construction exactly or the checkpoint will not load / the encoding drifts.
from beam.models.unet import ContextUnet
from beam.models.interpolant import ScoreMatch, EMA
from beam.models.probabilitypath import GaussianProbabilityPath, OTAlpha, OTBeta, VPBeta
from beam.models.simulator import ODE, SDE, ODEIntegrator, EulerMaruyama
from beam.utils.config import load_config, flatten_config
from beam.data.datasets import TESSDataset_angles_only

# Training files are (FFI - background) / 633118, scaled to a 30-minute exposure
# (1425.6 s effective), so this one factor converts them to e-/s for every cadence.
E_PER_S_PER_UNIT = 633118 / 1425.6


def sector_to_orbits(sector: int) -> set:
    """sector 1 = orbits 9,10"""
    return {2 * sector + 7, 2 * sector + 8}


def orbit_to_sector(orbit: int) -> int:
    return (int(orbit) - 7) // 2


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--sector",
        type=int,
        default=None,
    )
    group.add_argument(
        "--ffi-nums",
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--angle_path",
        type=str,
        default=None,
    )
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
    )
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only generate the first N selected FFIs",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--devices", type=str, nargs="+", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def resolve_settings(args):
    gen_config = flatten_config(load_config(args.config))

    def pick(cli_value, key, default=None):
        return cli_value if cli_value is not None else gen_config.get(key, default)

    # priority --devices > --device > config devices > config device
    if args.devices:
        devices = list(args.devices)
    elif args.device:
        devices = [args.device]
    else:
        cfg_devices = gen_config.get("generation_devices")
        devices = (
            list(cfg_devices)
            if cfg_devices
            else [gen_config.get("generation_device", "cuda:0")]
        )

    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError(
            f"--shard-index must be in [0, {args.num_shards}), got {args.shard_index}"
        )

    settings = argparse.Namespace(
        model_dir=pick(args.model_dir, "generation_model_dir"),
        checkpoint=pick(args.checkpoint, "generation_checkpoint"),
        output_dir=pick(args.output_dir, "generation_output_dir"),
        angle_path=pick(args.angle_path, "generation_angle_path"),
        num_steps=int(pick(args.num_steps, "generation_num_sde_steps", 500)),
        guidance_scale=float(
            pick(args.guidance_scale, "generation_guidance_scale", 1.0)
        ),
        batch_size=int(pick(args.batch_size, "generation_batch_size", 16)),
        devices=devices,
        seed=pick(args.seed, "generation_seed"),
        use_ema=bool(gen_config.get("generation_ema", True)) and not args.no_ema,
        sector=args.sector,
        ffi_nums=args.ffi_nums,
        limit=args.limit,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    if not settings.model_dir:
        raise ValueError(
            f"No model_dir: set generation.model_dir in {args.config} "
            "or pass --model_dir"
        )
    return settings


def find_checkpoint(model_dir, checkpoint_arg):
    if checkpoint_arg:
        path = (
            checkpoint_arg
            if os.path.isabs(checkpoint_arg) or os.path.exists(checkpoint_arg)
            else os.path.join(model_dir, checkpoint_arg)
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_arg}")
        return path
    finals = sorted(glob.glob(os.path.join(model_dir, "checkpoint_epoch_final_*.pth")))
    if finals:
        return finals[-1]
    checkpoints = glob.glob(os.path.join(model_dir, "checkpoint_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint_*.pth in {model_dir}")
    return max(checkpoints, key=os.path.getmtime)


def load_model(checkpoint_path, config, context_len, device, use_ema=True):
    unet = ContextUnet(
        in_feats=1,
        context_len=context_len,
        n_feat=config["model_n_feat"],
        channel_mults=config["model_channel_mults"],
        heads_at=config["model_heads_at"],
        time_dim=config.get("model_time_dim", 128),
        context_dim=config.get("model_context_dim", 64),
        emb_dim=config.get("model_emb_dim", 256),
        num_res=config["model_num_res"],
        use_cross_attn=config.get("model_use_cross_attn", True),
    )
    # Two schedules hardcoded
    model = ScoreMatch(
        nn_model=unet,
        probability_path=GaussianProbabilityPath(
            alpha=OTAlpha(),
            beta=OTBeta() if config["model_architecture"] == "flow" else VPBeta(),
        ),
        device=device,
        drop_prob=config.get("model_drop_prob", 0.1),
        epsilon=config.get("model_epsilon", 1e-4),
        architecture=config["model_architecture"],
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if use_ema:
        if "ema_state_dict" not in checkpoint:
            raise KeyError(
                f"No ema_state_dict in {checkpoint_path}; rerun with --no-ema"
            )
        ema = EMA(model=model)
        ema.load_state_dict(checkpoint["ema_state_dict"])
        ema.to(device)
        ema.copy_to(model)
        print(f"Loaded EMA weights from {checkpoint_path}")
    else:
        print(f"Loaded raw weights from {checkpoint_path}")

    model.eval()
    return model


def build_simulator(model, guidance_scale, num_steps, epsilon, device, cam=None):
    # cam: (B,) 0-based camera indices, one per sample in the batch
    if model.architecture == "flow":
        t = torch.linspace(epsilon, 1.0 - epsilon, 2, device=device).unsqueeze(0)
        return ODEIntegrator(ODE(model.nn_model, guidance_scale, cam=cam), t, num_save=1)
    t = torch.linspace(epsilon, 1.0 - epsilon, num_steps, device=device).unsqueeze(0)
    sde = SDE(
        nn_model=model.nn_model,
        probability_path=model.probability_path,
        guidance_value=guidance_scale,
        architecture=model.architecture,
        cam=cam,
    )
    return EulerMaruyama(sde, t, num_save=1)


def run_worker(worker_idx, run):
    entries = run["worker_entries"][worker_idx]
    if not entries:
        return
    device_str = run["devices"][worker_idx]
    args = run["args"]
    config = run["config"]

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    context_len = entries[0][2].shape[-1]
    model = load_model(
        run["checkpoint_path"], config, context_len, device, use_ema=args.use_ema
    )
    # same 0-based camera index the V2 dataset feeds the model during training
    cam_idx = int(run["camera"]) - 1

    if args.seed is not None:
        torch.manual_seed(int(args.seed) + 1000 * args.shard_index + worker_idx)

    output_dir = run["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    preview = []
    batches = range(0, len(entries), args.batch_size)
    for start in tqdm(batches, desc=f"[{device_str}] generating", position=worker_idx):
        chunk = entries[start : start + args.batch_size]
        c = torch.from_numpy(np.stack([ctx for (_, _, ctx) in chunk])).to(device)
        x0 = torch.randn(len(chunk), 1, *run["image_shape"], device=device)
        cam = torch.full((len(chunk),), cam_idx, dtype=torch.long, device=device)
        # rebuilt per chunk: cam must match the batch size (last chunk can be short)
        simulator = build_simulator(
            model, args.guidance_scale, args.num_steps, run["epsilon"], device, cam=cam
        )

        with torch.no_grad():
            xs, _ = simulator.simulate(x0, c)
        gen = xs[-1].float().cpu().numpy()[:, 0]  # (B, H, W)

        for i, (ffi, orbit, _) in enumerate(chunk):
            sector = orbit_to_sector(orbit)
            # denormalize, then convert to e-/s
            raw = (gen[i] * run["img_std"] + run["img_mean"]) * E_PER_S_PER_UNIT

            # potentially helpful metadata
            header = fits.Header()
            header["FFI"] = ffi
            header["ORBIT"] = orbit
            header["SECTOR"] = sector
            header["CAMERA"] = run["camera"]
            header["ARCH"] = model.architecture
            header["GUIDANCE"] = args.guidance_scale
            header["NSTEPS"] = (
                args.num_steps if model.architecture != "flow" else -1,
                "-1 = adaptive dopri5",
            )
            header["IMGMEAN"] = run["img_mean"]
            header["IMGSTD"] = run["img_std"]
            header["BUNIT"] = "electron/s"
            header["CKPT"] = os.path.basename(run["checkpoint_path"])
            header["EMA"] = args.use_ema

            fits_path = os.path.join(
                output_dir,
                f"generated_light_s{sector:04d}-{ffi}-cam{run['camera']}.fits",
            )
            fits.writeto(fits_path, raw.astype(np.float32), header, overwrite=True)

            if worker_idx == 0 and len(preview) < 16:
                preview.append((f"FFI {ffi} (o{orbit})", raw))

    if preview:
        ncols = min(4, len(preview))
        nrows = (len(preview) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        for ax, (title, img) in zip(np.atleast_1d(axes).ravel(), preview):
            ax.imshow(img, cmap="viridis", origin="lower")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        for ax in np.atleast_1d(axes).ravel()[len(preview) :]:
            ax.axis("off")
        label = f"sector {args.sector}" if args.sector is not None else "requested FFIs"
        fig.suptitle(f"Generated light — {label} (cam{run['camera']})", fontsize=12)
        plt.tight_layout()
        png_path = os.path.join(output_dir, "generation_preview.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"Saved preview {png_path}")

    print(f"[{device_str}] done: {len(entries)} generations")


def main():
    args = resolve_settings(parse_args())

    config = flatten_config(load_config(os.path.join(args.model_dir, "config.yaml")))
    # dataset stats computed from training run
    with open(os.path.join(args.model_dir, "context_stats.pkl"), "rb") as f:
        context_stats = pickle.load(f)
    with open(os.path.join(args.model_dir, "image_stats.pkl"), "rb") as f:
        image_stats = pickle.load(f)
    img_mean, img_std = float(image_stats["mean"]), float(image_stats["std"])
    print(f"Image stats: mean={img_mean:.6g}, std={img_std:.6g}")

    camera = str(config.get("data_camera_number", "3"))
    image_shape = tuple(config["data_image_shape"])
    epsilon = float(config.get("model_epsilon", 1e-4))

    angle_path = args.angle_path or config["data_angle_path"]
    angles_ds = TESSDataset_angles_only(
        angle_path=angle_path,
        camera_number=camera,
        context_stats=context_stats,
    )
    ffi_to_idx = {ffi: i for i, ffi in enumerate(angles_ds.ffi_nums)}
    if args.sector is not None:
        orbits = sector_to_orbits(args.sector)
        selected = [
            ffi
            for ffi in angles_ds.ffi_nums
            if int(angles_ds.angles_dic[ffi]["orbit"]) in orbits
        ]
        if not selected:
            raise ValueError(
                f"No FFIs for sector {args.sector} (orbits {sorted(orbits)}) "
                f"in {angle_path}"
            )
        print(f"Sector {args.sector} (orbits {sorted(orbits)}): {len(selected)} FFIs")
    else:
        selected, missing = [], []
        for ffi in args.ffi_nums:
            (selected if ffi in ffi_to_idx else missing).append(ffi)
        if missing:
            print(f"WARNING: FFIs not in angles dict, skipping: {missing}")
        if not selected:
            raise ValueError("None of the requested FFIs are in the angles dict")
        print(f"Generating for {len(selected)} requested FFIs")

    selected = sorted(selected)
    if args.limit is not None:
        selected = selected[: args.limit]
        print(f"Limiting to first {len(selected)} FFIs")

    if args.num_shards > 1:
        selected = selected[args.shard_index :: args.num_shards]
        print(f"Shard {args.shard_index}/{args.num_shards}: {len(selected)} FFIs")

    entries = [
        (
            ffi,
            int(angles_ds.angles_dic[ffi]["orbit"]),
            angles_ds[ffi_to_idx[ffi]]["x"].numpy(),
        )
        for ffi in selected
    ]
    devices = args.devices[: max(1, min(len(args.devices), len(entries)))]
    worker_entries = [entries[i :: len(devices)] for i in range(len(devices))]

    checkpoint_path = find_checkpoint(args.model_dir, args.checkpoint)
    sampler_name = (
        "ODE (dopri5)"
        if config["model_architecture"] == "flow"
        else f"SDE (Euler-Maruyama, {args.num_steps} steps)"
    )
    print(
        f"Architecture '{config['model_architecture']}' -> {sampler_name}, "
        f"guidance={args.guidance_scale}"
    )
    print(
        f"Devices: {devices} "
        f"({', '.join(str(len(w)) for w in worker_entries)} FFIs per worker)"
    )

    output_dir = args.output_dir or os.path.join(args.model_dir, "generated")
    os.makedirs(output_dir, exist_ok=True)

    run = {
        "args": args,
        "config": config,
        "devices": devices,
        "worker_entries": worker_entries,
        "checkpoint_path": checkpoint_path,
        "output_dir": output_dir,
        "img_mean": img_mean,
        "img_std": img_std,
        "camera": camera,
        "image_shape": image_shape,
        "epsilon": epsilon,
    }

    if len(devices) == 1:
        run_worker(0, run)
    else:
        mp.spawn(run_worker, args=(run,), nprocs=len(devices), join=True)

    print(f"\n{len(entries)} generations saved to {output_dir}")


if __name__ == "__main__":
    main()
