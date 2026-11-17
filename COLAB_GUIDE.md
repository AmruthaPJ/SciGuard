# Google Colab Setup Guide: Sci-Guard

Training on MacBook Air = **4 days**. Training on Colab T4 GPU = **~5–6 hours**. Use Colab.

---

## Before You Start

> **Stop any local training** — press `Ctrl+C` in your terminal right now.

1. Go to [colab.research.google.com](https://colab.research.google.com) → New Notebook
2. Set GPU: **Runtime → Change runtime type → T4 GPU → Save**

---

## Step 1: Get Your Kaggle API Key

The dataset is on Kaggle — download it directly inside Colab instead of uploading 5000 images.

1. Go to [kaggle.com](https://www.kaggle.com) → click your profile picture → **Settings**
2. Scroll to **API** section → click **Create New Token**
3. A file called `kaggle.json` will download to your Mac
4. Keep it handy — you'll upload it in the next step

---

## Step 2: Colab Setup Cell

Paste this into the **first cell** of your Colab notebook and run it:

```python
# ── 1. Upload your kaggle.json API key ──────────────────────────────────────
from google.colab import files
files.upload()  # Select your kaggle.json file when prompted

import os
os.makedirs('/root/.config/kaggle', exist_ok=True)
os.rename('kaggle.json', '/root/.config/kaggle/kaggle.json')
os.chmod('/root/.config/kaggle/kaggle.json', 0o600)

# ── 2. Download dataset directly from Kaggle (fast!) ────────────────────────
!pip install -q kaggle
!kaggle datasets download -d recodailuc/recodai-luc-scientific-image-forgery-detection -p /content/sifd --unzip

# ── 3. Install required libraries ───────────────────────────────────────────
!pip install -q timm albumentations tqdm scikit-learn

# ── 4. Verify folder structure ───────────────────────────────────────────────
!echo "=== Checking dataset structure ===" && ls /content/sifd/recodai-luc-scientific-image-forgery-detection/
```

Expected output:
```
train_images/   train_masks/   test_images/   ...
```

---

## Step 3: Run the Training

Paste this into a **new cell**:

```python
import os
# Tell the script where sifd lives in Colab
os.environ["SCIGUARD_DATA"] = "/content"
```

Then paste the **entire contents of `SciGuard_Complete.py`** into another new cell and run it.

> ✅ **No need to change `base_path`** — the code auto-detects it.
> But if it errors, manually set at the bottom of the code:
> ```python
> base_path = "/content/sifd"
> ```

---

## Step 4: Save Your Trained Weights

Training will save `best_sciguard.pth` automatically. Download it when done:

```python
from google.colab import files
files.download('best_sciguard.pth')
```

Then place `best_sciguard.pth` in your local `/Users/amrutha/Documents/SciGuard/` folder.
Your `app.py` backend will automatically pick it up.

---

## Step 5: Run the Dashboard Locally (with cloud-trained weights)

Back on your Mac:
```bash
python app.py          # Terminal 1 — starts backend at localhost:5001
open frontend/index.html  # Terminal 2 — opens dashboard
```

---

## Speed Comparison

| Where | Device | Time per Epoch | 30 Epochs Total |
|---|---|---|---|
| MacBook Air | Apple MPS | ~3 hours | **~4 days** 💀 |
| Google Colab | T4 GPU | ~10 min | **~5 hours** ✅ |
| Google Colab | A100 (Colab Pro) | ~3 min | **~90 min** ⚡ |

---

## Tips

- **Don't close the Colab tab** while training — it will disconnect after ~90 min of inactivity
- If disconnected, reload the page — Colab usually resumes the session
- Colab Pro ($10/month) gives you A100 GPU and no disconnects — worth it for this project
- The model saves `best_sciguard.pth` after every improvement, so even a partial run gives usable weights
