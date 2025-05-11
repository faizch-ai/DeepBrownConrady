
# DeepBrownConrady (DBC) – Camera-Calibration Parameter Prediction

**Author:** Faiz Chaudhry  
**Company:** AiLiveSim Ltd.

---

DeepBrownConrady (DBC) is a deep-learning framework that predicts camera intrinsics and Brown-Conrady distortion coefficients from images.  

It provides two main scripts:

- **`inference.py`** – Run the model, write a CSV of ground-truth vs prediction, and a timing file  
- **`evaluate_predictions.py`** – Compute normalized error metrics (MSE / RMSE / MAE) for H-FOV, norm-cx, norm-cy, and distortion terms

---

## 1. Project Layout

```
DeepBrownConrady/
├── DBC/                # datasets, models, utils
├── conda-env/          # dbc-env.yml, create_conda_env.sh
├── model/              # dbc.pth, optimizer_config.yml
├── data/               # optional sample images and JSONs
├── inference.py
├── evaluate_predictions.py
├── requirements.txt
└── README.md
```

---

## 2. Setup

### Step 1:

```bash
git clone https://github.com/your_username/DeepBrownConrady.git
```

### Step 2:

Follow the steps in [conda/README.md](conda/README.md) to create and activate the conda environment.

---

## 3. Running Inference

```bash
python inference.py     --folder_path ./data/kitti     --output_dir ./results/kitti     --model_path ./model/dbc.pth     --optim_path ./model/optimizer_config.yml     --model_type resnet50_extended_features     --batch_size 4     --num_workers 2     --scaling_factor 0.25     --device cuda
```

### Important Flags

| Flag               | Default                      | Explanation                            |
|--------------------|------------------------------|-------------------------------------- |
| `--folder_path (-p)` | **required**                 | Folder with images and matching JSON GT |
| `--output_dir (-o)`  | `results`                    | Directory for CSV + timing files       |
| `--model_path`       | `./model/dbc.pth`            | Pretrained weights                    |
| `--optim_path`       | `./model/optimizer_config.yml` | Normalization constants               |
| `--device`           | auto                         | `cuda` if available                    |
| `--batch_size`       | `1`                          | Larger batch size = faster on GPU     |
| `--scaling_factor`   | `0.25`                       | Resize factor before inference        |

### Outputs

- `<output_dir>/<Dataset>-predictions.csv`  
- `<output_dir>/<Dataset>-timing.txt`

---

## 4. Evaluating Predictions

```bash
python evaluate_predictions.py     --csv ./results/kitti-predictions.csv     --width 1392     --height 512
```

This will create a file named `kitti_normalized_error_metrics.csv` with MSE, RMSE, and MAE for:  
`hfov`, `norm_cx`, `norm_cy`, `k1`, `k2`, `k3`, `p1`, `p2`.

---

## 5. Sample Outputs

**Timing File (`kitti-timing.txt`):**

```
Dataset: Kitti
Total images: 512
Total inference time: 3.47 s
Average time per image: 0.0062 s
```

---

**Prediction CSV (`kitti-predictions.csv`, first row):**

```csv
image,fx_gt,fx_pred,fy_gt,fy_pred,cx_gt,cx_pred,cy_gt,cy_pred,k1_gt,k1_pred,k2_gt,k2_pred,k3_gt,k3_pred,p1_gt,p1_pred,p2_gt,p2_pred
Kitti001.png,984.2439,993.113,980.8141,993.113,690.0,686.52,233.1966,233.93,-0.37288,-0.37333,0.20373,0.21593,-0.07234,-0.07699,0.002219,0.003838,0.001384,0.001285
```

---

**Metric CSV (`kitti_normalized_error_metrics.csv`):**

```csv
parameter,MSE,RMSE,MAE
norm_cx,5.0e-06,0.0023,0.0018
norm_cy,8.0e-06,0.0028,0.0020
hfov,2.4e-05,0.0049,0.0037
k1,9.7e-05,0.0098,0.0079
k2,8.1e-05,0.0090,0.0074
k3,2.8e-05,0.0052,0.0038
p1,2.6e-05,0.0051,0.0031
p2,1.5e-05,0.0038,0.0029
```

---

## 6. License

MIT – see `LICENSE`.

---

## 7. Contact

Faiz Chaudhry · AiLiveSim Ltd.  
✉️ [faiz@ailivesim.com](mailto:faiz@ailivesim.com)