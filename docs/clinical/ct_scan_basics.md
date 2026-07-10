# CT Scan Basics for Liver Imaging

## What is a CT Scan?

Computed Tomography (CT) uses X-rays to create cross-sectional images of the body. Each pixel in a CT image represents the tissue density in Hounsfield Units (HU).

## Hounsfield Unit Scale

| Tissue | HU Range |
|--------|----------|
| Air | -1000 |
| Lung | -500 to -200 |
| Fat | -120 to -90 |
| Water | 0 |
| Liver (normal) | 50-70 |
| Kidney | 30-50 |
| Muscle | 10-40 |
| Blood | 40-60 |
| Bone | 400-1000 |

## CT Windowing

Windowing improves visualization of specific tissues:

| Window | Level (HU) | Width (HU) | Best For |
|--------|-----------|-----------|---------|
| Liver | 30 | 150 | Liver parenchyma |
| Abdomen | 50 | 400 | General abdominal survey |
| Bone | 400 | 1800 | Skeletal evaluation |
| Lung | -600 | 1500 | Pulmonary assessment |

## LiTS Dataset Context

- **Source**: LiTS (Liver Tumor Segmentation) 2017 Challenge
- **Resolution**: 512×512 pixels
- **Slices**: ~58,638 across 131 volumes
- **Ground truth**: Binary masks (liver+tumor = 1, background = 0)
- **Preprocessing applied**: HU windowing [-100, 400], CLAHE, resize to 256×256
