import numpy as np
from scipy import ndimage


class TumorBurdenAnalyzer:
    def __init__(self, voxel_spacing=(1.0, 1.0, 1.0)):
        self.voxel_spacing = voxel_spacing

    def tumor_volume(self, mask_slices, spacing=None):
        spacing = spacing or self.voxel_spacing
        pixel_volume = spacing[0] * spacing[1] * spacing[2]
        tumor_pixels = [np.sum(m == 2) for m in mask_slices]
        total_pixels = sum(tumor_pixels)
        return total_pixels * pixel_volume

    def tumor_surface_area(self, mask_3d, spacing=None):
        spacing = spacing or self.voxel_spacing
        binary = (mask_3d == 2).astype(np.uint8)
        surface_voxels = self._count_surface(binary)
        voxel_face_area = spacing[0] * spacing[1]
        return surface_voxels * voxel_face_area

    def _count_surface(self, binary_mask):
        struct = ndimage.generate_binary_structure(3, 1)
        eroded = ndimage.binary_erosion(binary_mask, struct)
        surface = binary_mask & ~eroded
        return np.sum(surface)

    def tumor_sphericity(self, mask_3d, spacing=None):
        from scipy import ndimage
        spacing = spacing or self.voxel_spacing
        binary = (mask_3d == 2).astype(np.uint8)
        volume = np.sum(binary) * spacing[0] * spacing[1] * spacing[2]
        labels, _ = ndimage.label(binary)
        if labels.max() == 0:
            return 0.0
        surface_area = self.tumor_surface_area(mask_3d, spacing)
        if surface_area == 0:
            return 0.0
        sphericity = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / surface_area
        return min(sphericity, 1.0)

    def tumor_location(self, mask_slices):
        tumor_present = [np.any(m == 2) for m in mask_slices]
        if not any(tumor_present):
            return "none"
        first = next(i for i, v in enumerate(tumor_present) if v)
        last = len(tumor_present) - next(i for i, v in enumerate(reversed(tumor_present)) if v)
        total = len(mask_slices)
        if first < total * 0.25:
            return "superior"
        elif last > total * 0.75:
            return "inferior"
        return "mid_volume"

    def per_slice_tumor_burden(self, mask_slices):
        return [np.sum(m == 2) for m in mask_slices]

    def summary(self, mask_slices, voxel_spacing=None):
        spacing = voxel_spacing or self.voxel_spacing
        vol = self.tumor_volume(mask_slices, spacing)
        stacked = np.stack(mask_slices) if isinstance(mask_slices, list) else mask_slices
        return {
            "tumor_volume_mm3": vol,
            "num_tumor_slices": sum(1 for m in mask_slices if np.any(m == 2)),
            "location": self.tumor_location(mask_slices),
        }
