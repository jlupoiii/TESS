"""
Dataset classes for TESS image data

Conditioning encoding feature layout: (16 features, standardized to zero mean / unit variance)
[1/ED, 1/MD, 1/ED^2, 1/MD^2, Eel,  sin(Eaz),  cos(Eaz), Mel,  sin(Maz),  cos(Maz), Ecam_el, sin(Ecam_az), cos(Ecam_az), Mcam_el, sin(Mcam_az), cos(Mcam_az)]
    followed (optionally) by the exposure flag and patch-location embedding.

 Exposure flag indicator value: 1.0 if orbit >= 61, -1.0 otherwise

Patch-location conditioning: row/col sin-cos embedding

"""

import os
import re
import time
import pickle
from typing import Dict, Tuple, Optional, Union, List

import numpy as np
import math
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from tqdm import tqdm


def build_context_features(angles: Dict, camera: str) -> np.ndarray:
    """
    12 parameter geometric info -> 16 features
    """
    e_az = angles['Eaz']
    m_az = angles['Maz']
    ec_az = angles['E' + camera + 'az']
    mc_az = angles['M' + camera + 'az']

    return np.array([
        angles['1/ED'], angles['1/MD'],
        angles['1/ED^2'], angles['1/MD^2'],
        angles['Eel'], math.sin(e_az), math.cos(e_az),
        angles['Mel'], math.sin(m_az), math.cos(m_az),
        angles['E' + camera + 'el'], math.sin(ec_az), math.cos(ec_az),
        angles['M' + camera + 'el'], math.sin(mc_az), math.cos(mc_az),
    ], dtype=np.float64)


CONTEXT_FEATURE_NAMES = [
    '1/ED', '1/MD', '1/ED^2', '1/MD^2',
    'Eel', 'sin(Eaz)', 'cos(Eaz)',
    'Mel', 'sin(Maz)', 'cos(Maz)',
    'Ecam_el', 'sin(Ecam_az)', 'cos(Ecam_az)',
    'Mcam_el', 'sin(Mcam_az)', 'cos(Mcam_az)',
]


class TESSDataset(Dataset):
    """
    Dataset for TESS (Transiting Exoplanet Survey Satellite) images and their
    corresponding orbital parameters.
    """
    def __init__(
        self,
        angle_path: str,
        ccd_folder: str,
        image_shape: Tuple[int, int],
        mean: float,
        std: float,
        camera_number: str,
        patch_size: Optional[Tuple[int, int]] = None,
        repeat_factor: int = 1,
        add_exposure_flag: bool = True,
        normalize_context: bool = True,
        sky_by_sector: Optional[Dict[int, float]] = None,
        zero_context: bool = False,
    ):
        start_time = time.time()
        # Define paths and parameters
        self.ccd_folders = list(ccd_folder) if isinstance(ccd_folder, (list, tuple)) else [ccd_folder]
        self.image_shape = image_shape
        self.angle_path = angle_path

        self.length = 0
        self.patch_size = patch_size
        self.repeat_factor = repeat_factor

        self.camera_number = camera_number

        self.add_exposure_flag = add_exposure_flag
        self.normalize_context = normalize_context
    
        self.zero_context = zero_context
        self.sky_by_sector = ({int(k): float(v) for k, v in sky_by_sector.items()}
                              if sky_by_sector is not None else None)
        self.context_mean: Optional[np.ndarray] = None
        self.context_std: Optional[np.ndarray] = None
        self.context_feature_names = list(CONTEXT_FEATURE_NAMES)

        if self.patch_size is not None:
            assert self.image_shape[0] % self.patch_size[0] == 0, "Image shape must be divisible by patch size"

            self.ph, self.pw = self.patch_size
            self.patch_x = self.image_shape[0] // self.pw
            self.patch_y = self.image_shape[1] // self.ph
            self.num_patches = self.patch_x * self.patch_y  # assume square images and patches so y = x
            self.embed_dim = 6
            self.row_embeds = embed_patch(torch.arange(self.patch_x), self.embed_dim)
            self.col_embeds = embed_patch(torch.arange(self.patch_y), self.embed_dim)

        # Load orbital parameter dictionary
        self.angles_dic = pickle.load(open(self.angle_path, "rb"))
        self.MEAN = mean
        self.STD = std

        # Find all valid image files that have corresponding angle data
        # store files for use in __getitem__
        self.files = []        # full image paths
        self.ffi_nums = []
        self.cameras = []
        self.sky_vals = [] if self.sky_by_sector is not None else None
        for folder in self.ccd_folders:
            for filename in os.listdir(folder):
                ffi_num, camera = parse_ffi_camera(filename)
                if ffi_num in self.angles_dic.keys():
                    if self.sky_by_sector is not None:
                        sector = (int(self.angles_dic[ffi_num]['orbit']) - 7) // 2
                        if sector not in self.sky_by_sector:
                            raise KeyError(
                                f"sky_by_sector has no entry for sector {sector} "
                                f"(file {filename}); add its sky constant")
                        self.sky_vals.append(self.sky_by_sector[sector])
                    self.files.append(os.path.join(folder, filename))
                    self.ffi_nums.append(ffi_num)
                    self.cameras.append(camera)
                    self.length += 1

        end_time = time.time()

        print(f"Indexed {self.length} candidate files (all orbits on disk) across "
              f"{len(self.ccd_folders)} folder(s) in {end_time - start_time:.2f} s. ")

    #now computing dataset stats at train time to make bookkeeping easier
    def compute_context_stats(self, indices: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        #use train indices
        if indices is None:
            indices = range(self.length)
        rows = np.stack([
            build_context_features(self.angles_dic[self.ffi_nums[i]], self.cameras[i])
            for i in indices
        ])
        mean = rows.mean(axis=0)
        std = rows.std(axis=0).clip(min=1e-6)
        return mean, std

    def set_context_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.context_mean = np.asarray(mean, dtype=np.float64)
        self.context_std = np.asarray(std, dtype=np.float64)

    def context_stats_dict(self) -> Dict:
        return {
            'mean': self.context_mean,
            'std': self.context_std,
            'feature_names': self.context_feature_names,
            'add_exposure_flag': self.add_exposure_flag,
        }

    def _get_mmap(self, path, cache):
        mm = cache.get(path)
        if mm is None:
            mm = np.load(path, mmap_mode='r', allow_pickle=True)
            cache[path] = mm
        return mm

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.length * self.repeat_factor

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Get a sample from the dataset.

        Returns dict of:
            x: Conditioning tensor of shape (1, L) 
            y: Image tensor
            ffi_num: FFI identification number
            orbit: Orbit number
            cam: 0-based camera index
        """
        file_path = self.files[idx]  # already a full path
        ffi_num = self.ffi_nums[idx]
        camera = self.cameras[idx]   # "1".."4", from this image's filename

        angles = self.angles_dic[ffi_num]
        orbit = angles['orbit']

        params = build_context_features(angles, camera)
        if self.normalize_context and self.context_mean is not None:
            params = (params - self.context_mean) / self.context_std
        context = torch.from_numpy(params.astype(np.float32))

        if self.add_exposure_flag:
            flag = 1.0 if int(orbit) >= 61 else -1.0
            context = torch.cat([context, torch.tensor([flag], dtype=torch.float32)])


        image_arr = np.load(file_path, mmap_mode='r', allow_pickle=True)

        # Apply patch if patch_size is not None
        if self.patch_size is not None:
            patch_idx = torch.randint(self.num_patches, (1,))
            patch_row, patch_col = divmod(patch_idx.item(), self.patch_x)
            top, left = patch_row * self.ph, patch_col * self.pw
            image_arr = image_arr[top:top + self.ph, left:left + self.pw]
            conditioning_loc = torch.cat([self.row_embeds[patch_row], self.col_embeds[patch_col]], dim=-1)
            context = torch.cat([context, conditioning_loc.to(torch.float32)])

        arr = np.asarray(image_arr, dtype=np.float32).reshape(
            self.image_shape if self.patch_size is None else self.patch_size)
        if self.sky_vals is not None:
            arr = arr - self.sky_vals[idx]     # sky is in pkl units
        ffi_image = (torch.from_numpy(arr).unsqueeze(0) - self.MEAN) / self.STD

        if self.zero_context:
            context = torch.zeros_like(context)

        return {
            "x": context.unsqueeze(0),   # Conditioning (1, L) — L discovered by the trainer
            "y": ffi_image,              # Image
            "ffi_num": ffi_num,          # FFI identification number
            "orbit": orbit,              # Orbit number
            "cam": torch.tensor(int(camera) - 1, dtype=torch.long),  # 0-based camera index
        }


class TESSDataset_angles_only(Dataset):
    """
    Conditioning-only dataset for generation. Uses stats computed at training to build same encoding as in training
    """
    def __init__(self, angle_path, camera_number: str = '2',
                 context_stats: Optional[Dict] = None):
        self.angles_dic = pickle.load(open(angle_path, "rb"))
        self.ffi_nums = list(self.angles_dic.keys())
        self.length = len(self.ffi_nums)
        self.camera_number = camera_number

        if context_stats is not None:
            self.context_mean = np.asarray(context_stats['mean'], dtype=np.float64)
            self.context_std = np.asarray(context_stats['std'], dtype=np.float64)
            self.add_exposure_flag = context_stats.get('add_exposure_flag', True)
        else:
            # Unnormalized output only valid against a model trained the same way.
            self.context_mean = None
            self.context_std = None
            self.add_exposure_flag = True

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        ffi_num = self.ffi_nums[idx]
        angles = self.angles_dic[ffi_num]

        params = build_context_features(angles, self.camera_number)
        if self.context_mean is not None:
            params = (params - self.context_mean) / self.context_std
        context = torch.from_numpy(params.astype(np.float32))

        if self.add_exposure_flag:
            flag = 1.0 if int(angles['orbit']) >= 61 else -1.0
            context = torch.cat([context, torch.tensor([flag], dtype=torch.float32)])

        return {
            "x": context.unsqueeze(0),
            "ffi_num": ffi_num,
        }

    def __len__(self) -> int:
        return self.length


_TICA_NAME = re.compile(r"hlsp_tica_tess_ffi_s\d+-(?:o\d+-)?(\d{8})-cam(\d)-ccd\d")


def parse_ffi_camera(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse the FFI number and camera from a TICA filename, returns None else
    """
    match = _TICA_NAME.match(filename)
    if match is None:
        return None, None
    return match.group(1), match.group(2)


def embed_patch(prow, embed_dim):
    """
    Embed a patch location into a vector of dimension embed_dim. (1 row and col embedding)
    """
    half = embed_dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half
    )
    prow = prow.unsqueeze(1)  # [N, 1]
    row_scaled = prow * freqs[None]  # [N, half]

    return torch.cat([
        torch.sin(row_scaled),
        torch.cos(row_scaled),
    ], dim=-1)


def parse_orbit_ranges(valid_orbits) -> List[Tuple[int, int]]:
    """
    Turn a list of orbit ranges into (first, last) pairs, both inclusive.
    Each entry can be '10-12', [10, 12] or a single orbit like 33.
    """
    ranges = []
    for item in valid_orbits:
        if isinstance(item, str):
            first, _, last = item.partition('-')
            first, last = int(first), int(last or first)
        elif isinstance(item, (list, tuple)):
            if len(item) != 2:
                raise ValueError(f"orbit range {item!r} must have two ends, e.g. [10, 12]")
            first, last = int(item[0]), int(item[1])
        else:
            first = last = int(item)
        if first > last:
            raise ValueError(f"orbit range {item!r} starts after it ends")
        ranges.append((first, last))
    return ranges


def create_train_valid_datasets_by_orbit(dataset: TESSDataset, valid_orbits):
    """
    Create training and validation datasets using orbit number as the split
    criterion.

    """
    ranges = parse_orbit_ranges(valid_orbits)
    in_valid = [
        any(first <= int(dataset.angles_dic[ffi_num]["orbit"]) <= last for first, last in ranges)
        for ffi_num in dataset.ffi_nums
    ]
    train_indices = [idx for idx, v in enumerate(in_valid) if not v]
    valid_indices = [idx for idx, v in enumerate(in_valid) if v]
    if not valid_indices:
        raise ValueError(f"no frames fall in the validation orbits {valid_orbits}")

    rep = dataset.repeat_factor
    print(f"Orbit split (validation orbits {', '.join(f'{a}-{b}' for a, b in ranges)}, the rest train): "
          f"{len(train_indices)} train / {len(valid_indices)} valid files"
          + (f"; repeat_factor x{rep} -> {len(train_indices)*rep} / {len(valid_indices)*rep} samples per epoch"
             if rep != 1 else ""))

    #compute training set stats
    if getattr(dataset, 'normalize_context', False) and len(train_indices) > 0:
        mean, std = dataset.compute_context_stats(train_indices)
        dataset.set_context_stats(mean, std)
        print("Conditioning stats (train orbits only):")
        for name, m, s in zip(dataset.context_feature_names, mean, std):
            print(f"  {name:>14s}: mean={m: .4e}  std={s: .4e}")

    train_dataset = Subset(dataset, train_indices * dataset.repeat_factor)
    valid_dataset = Subset(dataset, valid_indices * dataset.repeat_factor)
    return train_dataset, valid_dataset


class _RawFluxDataset(Dataset):
    """
    """
    def __init__(self, dataset: TESSDataset, indices):
        self.files = [dataset.files[i] for i in indices]
        self.skys = ([dataset.sky_vals[i] for i in indices]
                     if dataset.sky_vals is not None else None)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, k):
        fp = self.files[k]
        arr = np.asarray(np.load(fp, mmap_mode='r', allow_pickle=True),
                         dtype=np.float32)
        if self.skys is not None:
            arr = arr - self.skys[k]           # as in __getitem__
        return torch.from_numpy(arr.reshape(-1))


def compute_image_stats(dataset: TESSDataset, indices, device=None,
                        max_images=None, batch_size=16, num_workers=8):
    """ 
    """
    idx = sorted(set(int(i) for i in indices))
    if max_images is not None and len(idx) > int(max_images):
        sel = np.linspace(0, len(idx) - 1, int(max_images)).astype(int)
        idx = [idx[i] for i in sel]

    device = torch.device(device) if device is not None else \
        torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    loader = DataLoader(_RawFluxDataset(dataset, idx), batch_size=batch_size,
                        num_workers=num_workers, pin_memory=(device.type == 'cuda'))

    count = torch.zeros((), dtype=torch.float64, device=device)
    s1 = torch.zeros((), dtype=torch.float64, device=device)
    s2 = torch.zeros((), dtype=torch.float64, device=device)
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Computing image stats over {len(idx)} files"):
            b = batch.to(device, non_blocking=True).double()
            count += b.numel()
            s1 += b.sum()
            s2 += (b * b).sum()
    mean = (s1 / count).item()
    var = (s2 / count).item() - mean ** 2
    std = max(var, 1e-12) ** 0.5
    return float(mean), float(std)
