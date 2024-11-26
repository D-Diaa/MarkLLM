import os
from collections import defaultdict

import pandas as pd
#%%
folder = "evaluation/results"
evaluation_keys = ["None", "15485863"]
attack_name_map = {
    'Paraphrase(GPT3.5)': 'GPT3.5',
    'Paraphrase(GPT4o)': 'GPT4o',
    'Paraphrase(meta-llama/Llama-3.2-3B-Instruct)': 'Llama-3.2-3B-Instruct',
    'Paraphrase(Qwen/Qwen2.5-3B-Instruct)': 'Qwen2.5-3B-Instruct',
    'Paraphrase(models/Unigram_new/meta-llama/Llama-3.2-3B-Instruct)': 'Llama-3.2-3B-Instruct[Unigram]',
    'Paraphrase(models/Unigram_new/Qwen/Qwen2.5-3B-Instruct)': 'Qwen2.5-3B-Instruct[Unigram]',
    'Paraphrase(models/dpo_qwen_3_exponential)': 'Qwen2.5-3B-Instruct[EXP]',
    'none': 'Unmodified',
    'Word-D': 'Deletion',
    'Word-S': 'Substitution',
    'Word-S(Context)': 'Substitution(Context)',
}

skip_watermarks = ["DIP", "Unbiased"]

# File paths for each watermark dataset
file_paths = {}
for file in os.listdir(folder):
    if file.endswith(".json") and "_finaleval" in file:
        name = file.replace(".json", "").replace("finaleval", "")
        watermark, key = name.split("_")
        if key in evaluation_keys:
            file_paths[watermark] = os.path.join(folder, file)
#%%
# Initialize empty dictionary to store data for each metric
metrics_data = defaultdict(dict)

# Read each file and organize the data by metrics and attacks
for watermark, file_path in file_paths.items():
    if watermark in skip_watermarks:
        continue
    with open(file_path, 'r') as f:
        data = pd.read_json(f)
        for key, values in data.items():
            if key not in attack_name_map:
                continue
            attack = attack_name_map[key]
            for metric, value in values.items():
                if not isinstance(value, dict):
                    if watermark not in metrics_data[metric]:
                        metrics_data[metric][watermark] = {}
                    metrics_data[metric][watermark][attack] = value
                else:
                    for submetric in value:
                        if submetric not in metrics_data:
                            metrics_data[submetric] = {}
                        if watermark not in metrics_data[submetric]:
                            metrics_data[submetric][watermark] = {}
                        metrics_data[submetric][watermark][attack] = value[submetric]

# Convert metrics data into separate DataFrames for each metric
metric_tables = {metric: pd.DataFrame(data) for metric, data in metrics_data.items()}

# Save each metric table to a separate markdown file
for metric, table in metric_tables.items():
    table.to_markdown(f"evaluation/results/{metric}.md", tablefmt="github")
