import json
import os

from tqdm import tqdm

from evaluation.pipelines.robustness import WatermarkRobustnessPipeline

folder = "evaluation/results"
files = tqdm(list(os.listdir(folder)))
for file in files:
    if file.endswith(".tsv"):
        file_path = os.path.join(folder, file)
        stats = WatermarkRobustnessPipeline.get_summary_stats(filename=file_path)
        # remove tsv
        file_name = file.replace(".tsv", "")+".json"
        with open(os.path.join(folder, file_name), "w") as f:
            json.dump(stats, f)
