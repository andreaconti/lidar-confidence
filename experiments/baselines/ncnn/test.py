"""
Computes the AUC metrics for PNCNN baseline models
from "Uncertainty-Aware CNNs for Depth Completion: Uncertainty from Beginning to End"
(https://arxiv.org/abs/2006.03349)
"""

# % imports

import warnings
import torch
from lidar_confidence.dataset import Dataset
from lidar_confidence.metrics import (
    cost_curve_auc,
    cost_curve_optimal_auc,
    cost_curve_random_auc,
)
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from os import path
import sys
import json
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import wandb


# % load model name

model_name = sys.argv[1]
data_path = sys.argv[2]
results_path = sys.argv[3]

# % load model
common_path = path.join(path.dirname(__file__), "models")
code_path = path.join(path.dirname(__file__), "models", model_name)
model_path = path.join(code_path, "model.pth")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

sys.path.append(common_path)
sys.path.append(code_path)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = torch.load(model_path, map_location=device)["model"]


# % load data
def transform(d):
    for k in d.keys():
        d[k] = F.to_tensor(d[k])
    return d


kitti_completion = DataLoader(Dataset(data_path, split="test_1", transform=transform))
kitti_2015 = DataLoader(Dataset("data/dataset", split="test_2", transform=transform))

# % compute results
metrics = defaultdict(lambda: [])

print("computing AUC for kitti depth completion")
for batch in tqdm(kitti_completion):

    with torch.no_grad():
        conf = model(batch["lidar"].to(device)).cpu()[:, 2:]

    for crit in ["mae", "rmse"]:
        metrics[f"test_1/auc_{crit}"].append(
            cost_curve_auc(batch["gt"], batch["lidar"], conf, crit)
        )
        metrics[f"test_1/auc_opt_{crit}"].append(
            cost_curve_optimal_auc(batch["gt"], batch["lidar"], crit)
        )
        metrics[f"test_1/auc_rnd_{crit}"].append(
            cost_curve_random_auc(batch["gt"], batch["lidar"], crit)
        )

print("computing AUC for kitti 2015 (142 split)")
for batch in tqdm(kitti_2015):

    with torch.no_grad():
        conf = model(batch["lidar"].to(device)).cpu()[:, 2:]

    for crit in ["mae", "rmse"]:
        metrics[f"test_2/auc_{crit}"].append(
            cost_curve_auc(batch["gt"], batch["lidar"], conf, crit)
        )
        metrics[f"test_2/auc_opt_{crit}"].append(
            cost_curve_optimal_auc(batch["gt"], batch["lidar"], crit)
        )
        metrics[f"test_2/auc_rnd_{crit}"].append(
            cost_curve_random_auc(batch["gt"], batch["lidar"], crit)
        )

for k in metrics:
    metrics[k] = round(torch.stack(metrics[k]).mean().item(), 7)

# % save metrics in scores file

result_path = Path(results_path)
result_path.mkdir(exist_ok=True, parents=True)
scores_path = result_path / (model_name + ".json")
with open(scores_path.as_posix(), "w") as f:
    json.dump(metrics, f)

# % save metrics in wandb

wandb.init(
    name=model_name,
    tags=["baseline"],
    save_code=True,
    notes="see https://arxiv.org/abs/2006.03349",
)

metrics_wandb = {"/".join(n.split("/")[::-1]): v for n, v in metrics.items()}
wandb.log(metrics_wandb)
