# Sci-Guard: How to Run the Entire Project

Follow these steps to train the model and launch the Forensic Dashboard.

---

### Step 1: Install Dependencies
Make sure your Python environment has all the necessary libraries:
```bash
python -m pip install torch timm albumentations opencv-python-headless Flask Flask-Cors scikit-learn matplotlib
```

---

### Step 2: Training the Model (Required for Best Results)
The dashboard needs trained weights (`sciguard_final.pth`).
1.  Ensure your dataset is in the `sifd/` folder (as described in the README).
2.  Run the training script:
    ```bash
    python SciGuard_Complete.py
    ```
    *Note: Training on 5,000+ images will take several hours on a GPU.*

---

### Step 3: Launch the Forensic Backend
Once training is done (or using existing weights), start the Flask server:
```bash
python app.py
```
This will start the API at `http://localhost:5001`. Keep this terminal window open.

---

### Step 4: Launch the Frontend
In a **new** terminal window, open the dashboard:
```bash
open frontend/index.html
```

---

### 🚀 Automation Script
I have also created a `run_dashboard.sh` file. You can simply run:
```bash
sh run_dashboard.sh
```
This will automatically launch the backend and open the dashboard for you.

---

### 🔍 Troubleshooting
- **Port Error**: If you see `Address already in use`, it means port 5001 is taken. You can kill the previous process using `lsof -ti:5001 | xargs kill -9`.
- **Weights Warning**: If the dashboard shows "Low Risk" for every image, check if `best_sciguard_m1.pth` or `sciguard_final.pth` exists in the root folder.
