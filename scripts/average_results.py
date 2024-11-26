import json
import os

from tqdm import tqdm

#%%
folder = "evaluation/results"
watermark = "SIR"
keywords = ["new", ".json"]
files = [file for file in os.listdir(folder) if all(keyword in file for keyword in keywords) and watermark in file]
results = []
for file in tqdm(files):
    with open(os.path.join(folder, file), "r") as f:
        data = json.load(f)
    results.append(data)
#%%
output = {}
for data in results:
    for key, value in data.items():
        if key not in output:
            output[key] = []
        output[key].append(value)

#%%
averaged_output = {}
for attack, values_list in output.items():
    averaged_output[attack] = {}
    keys = values_list[0].keys()
    for metric in keys:
        if metric != "ratings":
            metric_values = [values[metric] for values in values_list]
            averaged_output[attack][metric] = sum(metric_values) / len(metric_values)
        else:
            rating_keys = values_list[0][metric].keys()
            averaged_output[attack][metric] = {}
            for rating_key in rating_keys:
                rating_values = [values[metric][rating_key] for values in values_list]
                averaged_output[attack][metric][rating_key] = sum(rating_values) / len(rating_values)
#%%
with open(os.path.join(folder, f"{watermark}_final.json"), "w") as f:
    json.dump(averaged_output, f)
