# MedSegX — Environment Snapshot

> **Date:** 2026-07-06
> **Purpose:** Reproducible record of the exact Python environment used for all MedSegX experiments.
> To recreate: `C:\Users\alanm\.local\bin\uv.exe sync` from project root (requires `pyproject.toml` with lock file).

---

## 1. System Overview

| Component | Value |
|-----------|-------|
| **OS** | Windows 11 |
| **GPU** | NVIDIA GeForce RTX 3050 Ti Laptop (4 GB VRAM) |
| **CUDA Driver** | 592.00 |
| **CUDA Runtime** | 13.1 |
| **Python** | 3.11.15 (managed by uv) |
| **uv** | `C:\Users\alanm\.local\bin\uv.exe` |
| **Venv location** | `D:\DATA SCIENCE AND ANALYTICS\PROJECTS\MedSegX\.venv\` |
| **Venv Python** | `D:\DATA SCIENCE AND ANALYTICS\PROJECTS\MedSegX\.venv\Scripts\python.exe` |
| **uv Python base** | `C:\Users\alanm\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\` |
| **Total packages** | 99 |

---

## 2. Core ML/DL Stack

| Package | Version | Path | Purpose |
|---------|---------|------|---------|
| **torch** | 2.5.1+cu124 | `.venv\Lib\site-packages\torch` | Deep learning framework — CUDA 12.4 build |
| **torchvision** | 0.20.1+cu124 | `.venv\Lib\site-packages\torchvision` | Pretrained backbones (MobileNetV2) |
| **numpy** | 1.26.4 | `.venv\Lib\site-packages\numpy` | Numerical computing (pinned <2) |
| **scipy** | 1.17.1 | `.venv\Lib\site-packages\scipy` | KDTree for surface metrics (HD95/ASD/NSD) |
| **scikit-learn** | 1.9.0 | `.venv\Lib\site-packages\sklearn` | Bootstrapping, train/test split utilities |
| **scikit-image** | 0.26.0 | `.venv\Lib\site-packages\skimage` | Image processing filters |
| **opencv-python** | 5.0.0.93 | `.venv\Lib\site-packages\cv2` | CLAHE, image resize, augmentation |
| **pillow** | 12.2.0 | `.venv\Lib\site-packages\PIL` | PNG image I/O |
| **matplotlib** | 3.11.0 | `.venv\Lib\site-packages\matplotlib` | Calibration plots, visualization |
| **pandas** | 2.3.3 | `.venv\Lib\site-packages\pandas` | DataFrame-based metrics aggregation |
| **nibabel** | 5.4.2 | `.venv\Lib\site-packages\nibabel` | NIfTI medical image format support |
| **tqdm** | 4.68.3 | `.venv\Lib\site-packages\tqdm` | Progress bars for training loops |
| **pyyaml** | 6.0.3 | `.venv\Lib\site-packages\yaml` | YAML config parsing |

---

## 3. Experiment Tracking & Web

| Package | Version | Purpose |
|---------|---------|---------|
| **mlflow** | 3.14.0 | Experiment tracking (params, metrics, artifacts) |
| **mlflow-skinny** | 3.14.0 | MLflow lightweight client |
| **mlflow-tracing** | 3.14.0 | MLflow distributed tracing |
| **streamlit** | 1.58.0 | Web dashboard for model inspection |
| **flask** | 3.1.3 | Lightweight API server |
| **fastapi** | 0.139.0 | Modern async API framework |
| **uvicorn** | 0.50.1 | ASGI server for FastAPI |
| **python-dotenv** | 1.2.2 | `.env` file loading for config |
| **pydeck** | 0.9.3 | Deck.gl visualization in Streamlit |

---

## 4. Profiling & Utilities

| Package | Version | Purpose |
|---------|---------|---------|
| **thop** | 0.1.1.post2209072238 | FLOPs and parameter counting |
| **psutil** (pywin32) | 312 | System resource monitoring |
| **pygments** | 2.20.0 | Code syntax highlighting |
| **prettytable** | 3.18.0 | Terminal-formatted tables |
| **gitpython** | 3.1.50 | Git integration for experiment tracking |

---

## 5. Testing

| Package | Version | Purpose |
|---------|---------|---------|
| **pytest** | 9.1.1 | Test framework |
| **iniconfig** | 2.3.0 | pytest config loading |
| **pluggy** | 1.6.0 | pytest plugin infrastructure |

---

## 6. Dependencies (transitive, auto-installed)

| Package | Version | Required By |
|---------|---------|-------------|
| aiohappyeyeballs | 2.7.1 | aiohttp |
| aiohttp | 3.14.1 | mlflow, streamlit |
| aiosignal | 1.4.0 | aiohttp |
| alembic | 1.18.5 | mlflow (DB migrations) |
| altair | 6.2.2 | streamlit (charts) |
| annotated-doc | 0.0.4 | streamlit |
| annotated-types | 0.7.0 | pydantic |
| anyio | 4.14.1 | starlette, httpcore |
| attrs | 26.1.0 | matplotlib, pytest |
| blinker | 1.9.0 | flask, streamlit |
| cachetools | 7.1.4 | google-auth |
| certifi | 2022.12.7 | requests |
| cffi | 2.0.0 | cryptography |
| charset-normalizer | 2.1.1 | requests |
| click | 8.4.2 | flask, streamlit, uvicorn |
| cloudpickle | 3.1.2 | streamlit, sklearn |
| colorama | 0.4.6 | Windows terminal colors |
| contourpy | 1.3.3 | matplotlib |
| cryptography | 48.0.1 | mlflow, databricks-sdk |
| cycler | 0.12.1 | matplotlib |
| databricks-sdk | 0.120.0 | mlflow (Databricks integration) |
| docker | 7.1.0 | mlflow (container deployment) |
| filelock | 3.29.0 | torch, torchvision |
| fonttools | 4.63.0 | matplotlib (font handling) |
| frozenlist | 1.8.0 | aiohttp |
| fsspec | 2026.4.0 | torch, mlflow (filesystem abstraction) |
| gitdb | 4.0.12 | gitpython |
| google-auth | 2.55.1 | mlflow (GCP integration) |
| graphene | 3.4.3 | mlflow (GraphQL API) |
| graphql-core | 3.2.11 | graphene |
| graphql-relay | 3.2.0 | graphene |
| greenlet | 3.5.3 | sqlalchemy (async support) |
| h11 | 0.16.0 | uvicorn (HTTP/1.1) |
| httptools | 0.8.0 | uvicorn (HTTP parsing) |
| huey | 3.1.1 | streamlit (task queue) |
| idna | 3.4 | requests, yarl |
| imageio | 2.37.3 | scikit-image |
| importlib-metadata | 9.0.0 | multiple packages |
| importlib-resources | 7.1.0 | matplotlib, jsonschema |
| itsdangerous | 2.2.0 | flask |
| jinja2 | 3.1.6 | flask, streamlit, matplotlib |
| joblib | 1.5.3 | sklearn |
| jsonschema | 4.26.0 | mlflow, streamlit |
| jsonschema-specifications | 2025.9.1 | jsonschema |
| kiwisolver | 1.5.0 | matplotlib |
| lazy-loader | 0.5 | scikit-image |
| mako | 1.3.12 | alembic |
| markupsafe | 3.0.3 | jinja2, mako |
| mpmath | 1.3.0 | sympy |
| multidict | 6.7.1 | aiohttp, yarl |
| narwhals | 2.23.0 | altair |
| networkx | 3.6.1 | scikit-image, torchvision |
| opentelemetry-api | 1.43.0 | mlflow (OpenTelemetry tracing) |
| opentelemetry-proto | 1.43.0 | opentelemetry |
| opentelemetry-sdk | 1.43.0 | opentelemetry |
| opentelemetry-semantic-conventions | 0.64b0 | opentelemetry |
| packaging | 26.2 | multiple packages |
| pluggy | 1.6.0 | pytest |
| propcache | 0.5.2 | aiohttp |
| protobuf | 6.33.6 | opentelemetry, mlflow |
| pyarrow | 24.0.0 | mlflow (parquet) |
| pyasn1 | 0.6.3 | pyasn1-modules, google-auth |
| pyasn1-modules | 0.4.2 | google-auth |
| pycparser | 3.0 | cffi |
| pydantic | 2.13.4 | mlflow, fastapi, streamlit |
| pydantic-core | 2.46.4 | pydantic |
| pygments | 2.20.0 | streamlit |
| pyparsing | 3.3.2 | matplotlib |
| python-dateutil | 2.9.0.post0 | matplotlib, pandas, streamlit |
| python-multipart | 0.0.32 | fastapi |
| pytz | 2026.2 | pandas |
| pywin32 | 312 | Windows COM/interop, jupyter |
| referencing | 0.37.0 | jsonschema |
| requests | 2.28.1 | mlflow, docker, torchvision |
| rpds-py | 2026.6.3 | jsonschema |
| six | 1.17.0 | python-dateutil |
| skops | 0.14.0 | sklearn model persistence |
| smmap | 5.0.3 | gitdb |
| sqlalchemy | 2.0.51 | mlflow (metadata store) |
| sqlparse | 0.5.5 | sqlalchemy |
| starlette | 1.3.1 | fastapi |
| sympy | 1.13.1 | torch (print_graph) |
| tenacity | 9.1.4 | mlflow (retry logic) |
| threadpoolctl | 3.6.0 | sklearn, opencv |
| tifffile | 2026.3.3 | scikit-image (TIFF I/O) |
| toml | 0.10.2 | streamlit, thop |
| typing-extensions | 4.15.0 | multiple packages |
| typing-inspection | 0.4.2 | pydantic |
| tzdata | 2026.2 | pandas (timezone) |
| urllib3 | 1.26.13 | requests |
| waitress | 3.0.2 | flask (production WSGI) |
| watchdog | 6.0.0 | streamlit (file watcher) |
| wcwidth | 0.8.2 | prettytable |
| websockets | 16.0 | uvicorn (WebSocket) |
| werkzeug | 3.1.8 | flask |
| yarl | 1.24.2 | aiohttp |
| zipp | 4.1.0 | importlib-metadata |

---

## 7. Version Pinning (for reproducibility)

To lock versions for future recreation, run from `MedSegX/`:

```powershell
C:\Users\alanm\.local\bin\uv.exe pip compile requirements.txt -o requirements-lock.txt
```

For `pyproject.toml`-based workflow:

```powershell
C:\Users\alanm\.local\bin\uv.exe lock
```

---

## 8. Installation Commands (if rebuilding)

```powershell
# Remove old venv
Remove-Item -Path ".venv" -Recurse -Force

# Create fresh venv with uv
C:\Users\alanm\.local\bin\uv.exe venv .venv --python 3.11

# Install from requirements.txt
C:\Users\alanm\.local\bin\uv.exe pip install -r requirements.txt

# Or with pyproject.toml
C:\Users\alanm\.local\bin\uv.exe sync
```

---

*Generated: 2026-07-06 | Tools: uv pip list, Python 3.11.15*
