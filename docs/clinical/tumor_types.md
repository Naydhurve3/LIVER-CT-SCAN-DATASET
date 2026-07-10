# Liver Tumor Types in CT Imaging

## Primary Liver Tumors

### Hepatocellular Carcinoma (HCC)
- **Prevalence**: ~75% of primary liver cancers
- **CT hallmarks**:
  - Arterial phase hyperenhancement
  - Washout in portal venous/delayed phases
  - Pseudocapsule enhancement
- **Risk factors**: Cirrhosis, Hepatitis B/C, alcohol, NAFLD
- **Segmentation challenge**: Heterogeneous appearance, irregular borders

### Cholangiocarcinoma
- **Origin**: Bile duct epithelium
- **CT appearance**: Delayed progressive enhancement
- **Types**: Mass-forming, periductal-infiltrating, intraductal

### Hepatoblastoma
- **Population**: Children (<3 years)
- **CT**: Heterogeneous mass with scattered calcifications

## Secondary (Metastatic) Liver Tumors

| Primary Site | CT Characteristics | Frequency |
|-------------|-------------------|-----------|
| Colon | Hypodense, rim enhancement | Very common |
| Breast | Variable appearance | Common |
| Lung | Often necrotic | Common |
| Pancreas | Hypodense | Common |
| Melanoma | Hyperdense (melanin) | Less common |

## Benign Liver Lesions

| Type | CT Features | Clinical Significance |
|------|------------|---------------------|
| Hemangioma | Peripheral nodular enhancement, fill-in on delayed | Incidental, no treatment needed |
| Simple Cyst | Water density, sharp borders, no enhancement | Incidental |
| FNH (Focal Nodular Hyperplasia) | Homogeneous, central scar | Incidental, "spoke-wheel" on angiography |
| Hepatic Adenoma | Hypervascular, may hemorrhage | Risk of malignant transformation |

## Clinical Relevance for Segmentation

1. **Binary segmentation** (liver+tumor vs background) is the standard approach
2. **Tumor heterogeneity** makes precise boundary detection challenging
3. **Small lesions** (<5mm) are easily missed in automated analysis
4. **Peritumoral edema** can mimic tumor infiltration
5. **Multi-focal disease** requires detection of all lesions, not just the largest
