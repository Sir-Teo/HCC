from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, List, Optional, Set, Tuple
import os
import warnings
import numpy as np
import pydicom
import nibabel as nib
from PIL import Image

from .extended import ExtendedVisionDataset

@dataclass
class _Entry:
    file_path: str
    modality: str  # 'dicom' or 'nifti'
    series_uid: str  # For DICOM series or NIfTI study grouping
    slice_index: Optional[int] = None  # For 3D volumes


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"

    @property
    def length(self) -> int:
        return {
            _Split.TRAIN: 0,  # To be set based on actual data
            _Split.VAL: 0,    # To be set based on actual data
        }[self]


class UnlabeledMedicalImageDataset(ExtendedVisionDataset):
    def __init__(
        self,
        *,
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        
        entries_path = self._get_entries_path(root)
        self._entries = self._load_extra(entries_path)
        
        series_ids_path = self._get_series_ids_path(root)
        self._series_ids = self._load_extra(series_ids_path)

    def _get_entries_path(self, root: Optional[str] = None) -> str:
        return "medical_entries.npy"

    def _get_series_ids_path(self, root: Optional[str] = None) -> str:
        return "series-ids.npy"

    def _find_series_ids(self, path: str) -> List[str]:
        series_ids = set()
        
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.dcm'):
                    # Extract series UID from DICOM
                    dcm = pydicom.dcmread(os.path.join(root, file))
                    series_ids.add(str(dcm.SeriesInstanceUID))
                elif file.endswith(('.nii', '.nii.gz')):
                    # Use filename as series ID for NIfTI
                    series_ids.add(os.path.splitext(file)[0])
                    
        return sorted(list(series_ids))

    def _load_entries_series_ids(self, root: Optional[str] = None) -> Tuple[List[_Entry], List[str]]:
        root = self.get_root(root)
        entries: List[_Entry] = []
        series_ids = self._find_series_ids(root)
        
        for root_dir, _, files in os.walk(root):
            for file in sorted(files):
                if file.endswith('.dcm'):
                    filepath = os.path.join(root_dir, file)
                    dcm = pydicom.dcmread(filepath)
                    entry = _Entry(
                        file_path=filepath,
                        modality='dicom',
                        series_uid=str(dcm.SeriesInstanceUID)
                    )
                    entries.append(entry)
                    
                elif file.endswith(('.nii', '.nii.gz')):
                    filepath = os.path.join(root_dir, file)
                    img = nib.load(filepath)
                    series_uid = os.path.splitext(file)[0]
                    
                    if img.ndim == 3:
                        # Create an entry for each slice in 3D volumes
                        for slice_idx in range(img.shape[2]):
                            entry = _Entry(
                                file_path=filepath,
                                modality='nifti',
                                series_uid=series_uid,
                                slice_index=slice_idx
                            )
                            entries.append(entry)
                    else:
                        entry = _Entry(
                            file_path=filepath,
                            modality='nifti',
                            series_uid=series_uid
                        )
                        entries.append(entry)
                        
        return entries, series_ids

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_full_path = os.path.join(self._extra_root, extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        extra_full_path = os.path.join(self._extra_root, extra_path)
        os.makedirs(self._extra_root, exist_ok=True)
        np.save(extra_full_path, extra_array)

    def _normalize_and_convert_to_rgb(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to range [0, 255] and convert to RGB."""
        img = img - img.min()
        if img.max() != 0:
            img = (img / img.max() * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
            
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img

    def get_image_data(self, index: int) -> np.ndarray:
        entry = self._entries[index]
        
        try:
            if entry['modality'] == 'dicom':
                dcm = pydicom.dcmread(entry['file_path'])
                img = dcm.pixel_array
                
            else:  # nifti
                nii = nib.load(entry['file_path'])
                img_data = nii.get_fdata()
                
                if img_data.ndim == 3:
                    slice_idx = entry['slice_index']
                    img = img_data[:, :, slice_idx]
                else:
                    img = img_data
                    
            img = self._normalize_and_convert_to_rgb(img)
            
        except Exception as e:
            raise RuntimeError(f"Cannot retrieve image data for sample {index} "
                             f"from {entry['file_path']}") from e
            
        return img

    def get_series_id(self, index: int) -> str:
        return str(self._entries[index]['series_uid'])

    def get_series_ids(self) -> np.ndarray:
        return self._entries['series_uid']

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return super().__getitem__(index)

    def __len__(self) -> int:
        return len(self._entries)

    def _dump_entries(self, *args, **kwargs) -> None:
        entries, series_ids = self._load_entries_series_ids(*args, **kwargs)

        max_path_length = max(len(entry.file_path) for entry in entries)
        max_series_id_length = max(len(str(entry.series_uid)) for entry in entries)

        dtype = np.dtype([
            ('file_path', f'U{max_path_length}'),
            ('modality', 'U6'),  # 'dicom' or 'nifti'
            ('series_uid', f'U{max_series_id_length}'),
            ('slice_index', '<i4'),
        ])

        entries_array = np.empty(len(entries), dtype=dtype)
        for i, entry in enumerate(entries):
            entries_array[i] = (
                entry.file_path,
                entry.modality,
                entry.series_uid,
                entry.slice_index if entry.slice_index is not None else -1
            )

        entries_path = self._get_entries_path(*args, **kwargs)
        self._save_extra(entries_array, entries_path)

    def _dump_series_ids(self, *args, **kwargs) -> None:
        entries_path = self._get_entries_path(*args, **kwargs)
        entries_array = self._load_extra(entries_path)

        unique_series_ids = np.unique(entries_array['series_uid'])
        series_ids_path = self._get_series_ids_path(*args, **kwargs)
        self._save_extra(unique_series_ids, series_ids_path)

    def _dump_extra(self, *args, **kwargs) -> None:
        self._dump_entries(*args, **kwargs)
        self._dump_series_ids(*args, **kwargs)

    def dump_extra(self, root: Optional[str] = None) -> None:
        return self._dump_extra(root)