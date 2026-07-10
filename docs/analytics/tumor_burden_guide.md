# Tumor Burden Analysis Guide

## What is Tumor Burden?

Tumor burden is a quantitative measure of the amount of tumor in a patient. In CT analysis, it's typically expressed as:

- **Tumor volume** (mm³ or cm³)
- **Number of lesions**
- **Percentage of organ involved**
- **Sum of longest diameters** (RECIST criteria)

## Metrics in This Project

### 1. Tumor Volume
- Calculated by summing tumor pixels and multiplying by voxel volume
- Voxel volume = spacing_x × spacing_y × spacing_z
- Clinical relevance: Correlates with tumor stage (T-stage in TNM)

### 2. Tumor Surface Area
- Approximated using Sobel edge detection on the mask
- Higher surface-to-volume ratio suggests more irregular/invasive tumors

### 3. Sphericity
- How round the tumor is: 1.0 = perfect sphere
- High sphericity (>0.8): Well-circumscribed, likely benign or early HCC
- Low sphericity (<0.5): Irregular, potentially infiltrative

### 4. Tumor Location
- Center of mass coordinates within the volume
- Spread (standard deviation) indicates how diffuse the tumor is

### 5. Per-Slice Coverage
- Percentage of each slice occupied by tumor
- Useful for identifying the "core" of the lesion vs. peripheral involvement

## Trend Analysis

The tumor burden trend across slices within a single scan can indicate:

| Trend | Interpretation |
|-------|---------------|
| Increasing | Tumor larger in inferior/superior portions — 3D growth pattern |
| Decreasing | Tumor concentrated in one region |
| Stable | Uniform involvement |
| Bimodal | Possible multiple separate lesions |

## Clinical Application

- **Treatment monitoring**: Compare tumor burden across serial scans
- **Prognosis**: Higher burden correlates with worse outcomes
- **Surgical planning**: Location and extent inform resectability
- **Tumor staging**: TNM classification uses size and number criteria
