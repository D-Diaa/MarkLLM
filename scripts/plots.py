# Load the data from the JSON file
import json
import os

import matplotlib.pyplot as plt
from tqdm import tqdm

folder = "evaluation/results"
files = tqdm(list(os.listdir(folder)))
for file in files:
    if file.endswith(".json"):
        with open(os.path.join(folder, file), "r") as f:
            data = json.load(f)
        file = file.replace(".json", "")
        # Prepare data for scatter plot
        x = [1 - model_data["default_detection"] for model_data in data.values()]
        y = [model_data["ratings"]["llm_judge"] for model_data in data.values()]
        labels = list(data.keys())
        # Correct label mapping for all entries in the data
        final_group_styles = {
            # GPTs with same color but different symbols
            "Paraphrase(GPT3.5)": ('o', 'blue'),
            "Paraphrase(GPT4o)": ('s', 'blue'),

            # Unigram with same color but different symbols
            "Paraphrase(models/Unigram/meta-llama/Llama-3.2-3B-Instruct)": ('o', 'green'),
            "Paraphrase(models/Unigram/Qwen/Qwen2.5-3B-Instruct)": ('s', 'green'),

            # Unigram_New with same color but different symbols
            "Paraphrase(models/Unigram_new/meta-llama/Llama-3.2-3B-Instruct)": ('o', 'orange'),
            "Paraphrase(models/Unigram_new/Qwen/Qwen2.5-3B-Instruct)": ('s', 'orange'),

            # DPO as its own class
            "Paraphrase(models/dpo_qwen_3_exponential)": ('*', 'purple'),

            # Other Paraphrase models in the same color but different symbols
            "Paraphrase(meta-llama/Llama-3.1-8B-Instruct)": ('^', 'red'),
            "Paraphrase(Qwen/Qwen2.5-3B-Instruct)": ('v', 'red'),

            # None as its own class
            "none": ('X', 'black'),

            # Word models with same color but different symbols
            "Word-D": ('P', 'brown'),
            "Word-S": ('H', 'brown'),
        }

        # Scatter plot with the corrected label mapping
        plt.figure(figsize=(12, 8))
        for i in range(len(labels)):
            label = labels[i]
            if label in final_group_styles:
                symbol, color = final_group_styles[label]
                plt.scatter(x[i], y[i], marker=symbol, color=color, label=label if label not in plt.gca().get_legend_handles_labels()[1] else "")

        plt.title(f"{file}")
        plt.xlabel("Attack Success")
        plt.ylabel("Rating")
        plt.legend(title="Models", fontsize="small", loc="best")
        plt.grid(True)
        plt.savefig(os.path.join(folder, f"{file}.png"))
        plt.clf()
        plt.close()
