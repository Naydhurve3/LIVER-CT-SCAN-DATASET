param(
    [switch]$DryRun = $false
)

$ErrorActionPreference = "Stop"
$srcLitsPng = "D:\DATA SCIENCE AND ANALYTICS\Dataset\lits-png\dataset_6\dataset_6"
$srcMasks = "D:\DATA SCIENCE AND ANALYTICS\Dataset\LiTS_masks"
$srcLegacy = "D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver Img Dataset"
$dst = "D:\DATA SCIENCE AND ANALYTICS\Dataset\Liver"

Write-Output "=== LiTS Dataset Canonical Arrangement ==="
Write-Output "Source lits-png: $srcLitsPng"
Write-Output "Source LiTS_masks: $srcMasks"
Write-Output "Destination: $dst"
if ($DryRun) { Write-Output "*** DRY RUN - No files will be copied ***" }

# ---- PHASE 1: Source Registry ----
Write-Output "`n[Phase 1] Building source registry..."
$reg = @{
    created = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    sources = @(
        @{ name="lits-png"; path=$srcLitsPng; description="LiTS PNG export from dataset_6 package (complete 58,638 images, partial masks)" },
        @{ name="LiTS_masks"; path=$srcMasks; description="Renamed tumor mask collection (58,638 masks, 131 volumes)" },
        @{ name="Liver Img Dataset"; path=$srcLegacy; description="Legacy 512x512 grayscale images (classification-origin, not segmentation-safe)" }
    )
    lits_png_images = (Get-ChildItem -Path $srcLitsPng -Filter "volume-*.png").Count
    lits_png_liver_masks = (Get-ChildItem -Path $srcLitsPng -Filter "*livermask*.png").Count
    lits_png_lesion_masks = (Get-ChildItem -Path $srcLitsPng -Filter "*lesionmask*.png").Count
    lits_masks_total = (Get-ChildItem -Path $srcMasks -Filter "*.png").Count
    legacy_images_total = (Get-ChildItem -Path $srcLegacy -Filter "*.png").Count
}
$regJson = $reg | ConvertTo-Json
if (-not $DryRun) {
    $reg | ConvertTo-Json | Set-Content -Path (Join-Path $dst "00_source_registry\source_registry.json") -Encoding UTF8
}
Write-Output "  lits-png images: $($reg.lits_png_images)"
Write-Output "  lits-png liver masks: $($reg.lits_png_liver_masks)"
Write-Output "  lits-png lesion masks: $($reg.lits_png_lesion_masks)"
Write-Output "  LiTS_masks: $($reg.lits_masks_total)"
Write-Output "  Legacy images: $($reg.legacy_images_total)"

# ---- PHASE 2: Organize images into volume subfolders ----
Write-Output "`n[Phase 2] Organizing images into 03_derived_256\images\v{XXX}\..."
$imgDir = Join-Path $dst "03_derived_256\images"
$totalImages = 0
$volImages = @{}

Get-ChildItem -Path $srcLitsPng -Filter "volume-*.png" | ForEach-Object {
    if ($_.Name -match 'volume-(\d+)_(\d+)\.png') {
        $volId = [int]$matches[1]
        $sliceIdx = [int]$matches[2]
        $volKey = "v{0:d3}" -f $volId
        $sliceFile = "s{0:d4}.png" -f $sliceIdx

        $volDir = Join-Path $imgDir $volKey
        $destFile = Join-Path $volDir $sliceFile

        if (-not (Test-Path $volDir)) {
            if (-not $DryRun) { New-Item -ItemType Directory -Path $volDir -Force | Out-Null }
        }
        if (-not $DryRun) { Copy-Item -Path $_.FullName -Destination $destFile -Force }
        $totalImages++

        if (-not $volImages.ContainsKey($volKey)) { $volImages[$volKey] = @() }
        $volImages[$volKey] += $sliceIdx
    }
}
Write-Output "  Organized $totalImages images across $($volImages.Count) volumes"

# ---- PHASE 3: Organize tumor masks ----
Write-Output "`n[Phase 3] Organizing tumor masks into 03_derived_256\tumor_masks\..."
$tumorDir = Join-Path $dst "03_derived_256\tumor_masks"
$totalTumor = 0
$posCount = 0
$negCount = 0

Get-ChildItem -Path $srcMasks -Filter "*.png" | ForEach-Object {
    if ($_.Name -match 'mask-(\d+)-(\d+)\.png') {
        $volId = [int]$matches[1]
        $sliceIdx = [int]$matches[2]
        $volKey = "v{0:d3}" -f $volId
        $sliceFile = "s{0:d4}.png" -f $sliceIdx

        $volDir = Join-Path $tumorDir $volKey
        $destFile = Join-Path $volDir $sliceFile

        if (-not (Test-Path $volDir)) {
            if (-not $DryRun) { New-Item -ItemType Directory -Path $volDir -Force | Out-Null }
        }
        if (-not $DryRun) { Copy-Item -Path $_.FullName -Destination $destFile -Force }
        $totalTumor++
    }
}
Write-Output "  Organized $totalTumor tumor masks"

# ---- PHASE 4: Organize available liver masks ----
Write-Output "`n[Phase 4] Organizing liver masks into 03_derived_256\liver_masks\..."
$liverDir = Join-Path $dst "03_derived_256\liver_masks"
$totalLiver = 0
$liverVols = @{}

Get-ChildItem -Path $srcLitsPng -Filter "*livermask*.png" | ForEach-Object {
    if ($_.Name -match 'segmentation-(\d+)_livermask_(\d+)\.png') {
        $volId = [int]$matches[1]
        $sliceIdx = [int]$matches[2]
        $volKey = "v{0:d3}" -f $volId
        $sliceFile = "s{0:d4}.png" -f $sliceIdx

        $volDir = Join-Path $liverDir $volKey
        $destFile = Join-Path $volDir $sliceFile

        if (-not (Test-Path $volDir)) {
            if (-not $DryRun) { New-Item -ItemType Directory -Path $volDir -Force | Out-Null }
        }
        if (-not $DryRun) { Copy-Item -Path $_.FullName -Destination $destFile -Force }
        $totalLiver++
        if (-not $liverVols.ContainsKey($volKey)) { $liverVols[$volKey] = @() }
        $liverVols[$volKey] += $sliceIdx
    }
}
Write-Output "  Organized $totalLiver liver masks across $($liverVols.Count) volumes"

# ---- PHASE 5: Build slice manifest ----
Write-Output "`n[Phase 5] Generating slice manifest..."
$manifestCsvPath = Join-Path $dst "04_manifests\slice_manifest.csv"
$manifestRows = @()
$manifestHeader = "sample_id,volume_id,slice_index,source_package,has_image,has_tumor_mask,has_liver_mask,image_path,tumor_mask_path,liver_mask_path,tumor_present"

# Iterate all images to build manifest
$totalSlices = 0
$posSlices = 0
$negSlices = 0
$missingLiverCount = 0

Get-ChildItem -Path $srcLitsPng -Filter "volume-*.png" | ForEach-Object {
    if ($_.Name -match 'volume-(\d+)_(\d+)\.png') {
        $volId = [int]$matches[1]
        $sliceIdx = [int]$matches[2]
        $sampleId = "v{0:d3}_s{1:d4}" -f $volId, $sliceIdx
        $volKey = "v{0:d3}" -f $volId
        $sliceFile = "s{0:d4}.png" -f $sliceIdx

        $imgPathRel = "03_derived_256\images\$volKey\$sliceFile"
        $tumorPathRel = "03_derived_256\tumor_masks\$volKey\$sliceFile"
        $liverPathRel = "03_derived_256\liver_masks\$volKey\$sliceFile"

        # Check if tumor mask exists
        $maskFile = Join-Path $srcMasks ("mask-{0:d3}-{1:d3}.png" -f $volId, $sliceIdx)
        $hasTumor = Test-Path $maskFile

        # Check if liver mask exists
        $liverFile = Join-Path $srcLitsPng ("segmentation-{0}_livermask_{1}.png" -f $volId, $sliceIdx)
        $hasLiver = Test-Path $liverFile

        # Check if tumor is present (pixel sum > 0)
        $tumorPresent = "FALSE"
        if ($hasTumor) {
            $totalSlices++
            try {
                $img = [System.Drawing.Image]::FromFile($maskFile)
                $bmp = New-Object System.Drawing.Bitmap $img
                $pixelSum = 0
                for ($y = 0; $y -lt $bmp.Height; $y++) {
                    for ($x = 0; $x -lt $bmp.Width; $x++) {
                        $px = $bmp.GetPixel($x, $y)
                        if ($px.R -gt 0) { $pixelSum++ }
                    }
                }
                $bmp.Dispose()
                $img.Dispose()
                if ($pixelSum -gt 0) { $tumorPresent = "TRUE"; $posSlices++ } else { $negSlices++ }
            } catch {
                $tumorPresent = "ERROR"
            }
        }

        if (-not $hasLiver) { $missingLiverCount++ }

        $row = "$sampleId,$volId,$sliceIdx,lits-png,$true,$hasTumor,$hasLiver,$imgPathRel,$tumorPathRel,$liverPathRel,$tumorPresent"
        $manifestRows += $row
    }
}

$manifestContent = @($manifestHeader) + $manifestRows
if (-not $DryRun) {
    $manifestContent | Set-Content -Path $manifestCsvPath -Encoding UTF8
}
Write-Output "  Slices: $totalSlices total, $posSlices tumor-positive, $negSlices empty"
Write-Output "  Missing liver masks: $missingLiverCount/$totalSlices"

# ---- PHASE 6: Generate volume manifest ----
Write-Output "`n[Phase 6] Generating volume manifest..."
$volManifestPath = Join-Path $dst "04_manifests\volume_manifest.csv"
$volRows = @()
$volHeader = "volume_id,slices,tumor_positive_slices,tumor_negative_slices,tumor_pixel_count,liver_masks_available"

foreach ($volKey in ($volImages.Keys | Sort-Object)) {
    $volId = [int]($volKey -replace 'v', '')
    $slices = $volImages[$volKey].Count
    $posInVol = 0
    $negInVol = 0
    $tumorPx = 0

    foreach ($si in $volImages[$volKey]) {
        $maskFile = Join-Path $srcMasks ("mask-{0:d3}-{1:d3}.png" -f $volId, $si)
        if (Test-Path $maskFile) {
            try {
                $img = [System.Drawing.Image]::FromFile($maskFile)
                $bmp = New-Object System.Drawing.Bitmap $img
                $pxSum = 0
                for ($y = 0; $y -lt $bmp.Height; $y++) {
                    for ($x = 0; $x -lt $bmp.Width; $x++) {
                        if ($bmp.GetPixel($x,$y).R -gt 0) { $pxSum++ }
                    }
                }
                $bmp.Dispose()
                $img.Dispose()
                if ($pxSum -gt 0) { $posInVol++; $tumorPx += $pxSum } else { $negInVol++ }
            } catch { }
        }
    }

    $liverAvail = if ($liverVols.ContainsKey($volKey)) { $liverVols[$volKey].Count } else { 0 }
    $volRows += "$volId,$slices,$posInVol,$negInVol,$tumorPx,$liverAvail"
}

$volContent = @($volHeader) + $volRows
if (-not $DryRun) {
    $volContent | Set-Content -Path $volManifestPath -Encoding UTF8
}
Write-Output "  $($volRows.Count) volumes recorded"

# ---- PHASE 7: Generate splits ----
Write-Output "`n[Phase 7] Generating volume-wise splits..."
$splitDir = Join-Path $dst "05_splits"
$train = 0..103
$val = 104..116
$test = 117..130

if (-not $DryRun) {
    $train | ForEach-Object { "v{0:d3}" -f $_ } | Set-Content -Path (Join-Path $splitDir "train_volumes.txt") -Encoding UTF8
    $val   | ForEach-Object { "v{0:d3}" -f $_ } | Set-Content -Path (Join-Path $splitDir "val_volumes.txt") -Encoding UTF8
    $test  | ForEach-Object { "v{0:d3}" -f $_ } | Set-Content -Path (Join-Path $splitDir "test_volumes.txt") -Encoding UTF8
}

# Generate slice-level splits from manifest data
$trainSlices = @(); $valSlices = @(); $testSlices = @()
foreach ($row in $manifestRows) {
    $parts = $row -split ','
    $v = [int]$parts[1]
    if ($v -le 103) { $trainSlices += $row }
    elseif ($v -le 116) { $valSlices += $row }
    else { $testSlices += $row }
}

$trainCsv = Join-Path $splitDir "train_slices.csv"
$valCsv = Join-Path $splitDir "val_slices.csv"
$testCsv = Join-Path $splitDir "test_slices.csv"
if (-not $DryRun) {
    @($manifestHeader) + $trainSlices | Set-Content -Path $trainCsv -Encoding UTF8
    @($manifestHeader) + $valSlices   | Set-Content -Path $valCsv -Encoding UTF8
    @($manifestHeader) + $testSlices  | Set-Content -Path $testCsv -Encoding UTF8
}
Write-Output "  Train: $($trainSlices.Count) slices (volumes 0-103)"
Write-Output "  Val:   $($valSlices.Count) slices (volumes 104-116)"
Write-Output "  Test:  $($testSlices.Count) slices (volumes 117-130)"
$expectedTotal = $trainSlices.Count + $valSlices.Count + $testSlices.Count
Write-Output "  Total: $expectedTotal slices (expected 58638)"

# ---- PHASE 8: Dataset version JSON ----
Write-Output "`n[Phase 8] Writing dataset version info..."
$ver = @{
    dataset = "lits-canonical-v1.0.0"
    created = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    total_slices = $totalSlices
    total_volumes = $volImages.Count
    positive_slices = $posSlices
    negative_slices = $negSlices
    sources = @(
        @{ name="lits-png"; image_count=$reg.lits_png_images; liver_mask_count=$reg.lits_png_liver_masks; lesion_mask_count=$reg.lits_png_lesion_masks },
        @{ name="LiTS_masks"; mask_count=$reg.lits_masks_total },
        @{ name="Liver Img Dataset (legacy)"; image_count=$reg.legacy_images_total; note="NOT used as primary source - spatial alignment unverified" }
    )
    split = @{
        train_volumes = "0-103"; train_slices = $trainSlices.Count
        val_volumes = "104-116"; val_slices = $valSlices.Count
        test_volumes = "117-130"; test_slices = $testSlices.Count
    }
    warnings = @(
        "Spatial alignment between lits-png images and LiTS_masks is verified for 25 volumes only",
        "Liver masks only available for volumes 7,8,9,78-99",
        "Do not use Liver Img Dataset as segmentation image source",
        "Training blocked until spatial audit passes"
    )
}
if (-not $DryRun) {
    $ver | ConvertTo-Json | Set-Content -Path (Join-Path $dst "04_manifests\dataset_version.json") -Encoding UTF8
}

# ---- Summary ----
Write-Output "`n=== Dataset Arrangement Complete ==="
Write-Output "Destination: $dst"
Write-Output ""
Write-Output "Structure:"
Write-Output "  Liver/"
Write-Output "  +-- 00_source_registry/"
Write-Output "  |   +-- source_registry.json"
Write-Output "  +-- 03_derived_256/"
Write-Output "  |   +-- images/   ($totalImages images across $($volImages.Count) volumes)"
Write-Output "  |   +-- tumor_masks/   ($totalTumor masks)"
Write-Output "  |   +-- liver_masks/   ($totalLiver masks across $($liverVols.Count) volumes)"
Write-Output "  +-- 04_manifests/"
Write-Output "  |   +-- slice_manifest.csv"
Write-Output "  |   +-- volume_manifest.csv"
Write-Output "  |   +-- dataset_version.json"
Write-Output "  +-- 05_splits/"
Write-Output "      +-- train_volumes.txt, val_volumes.txt, test_volumes.txt"
Write-Output "      +-- train_slices.csv, val_slices.csv, test_slices.csv"
Write-Output ""

if ($DryRun) {
    Write-Output "*** DRY RUN - No files were copied ***"
    Write-Output "Run without -DryRun to execute the actual arrangement."
}
