# Load the data from the JSON file
import json
import os
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm
#%%
plt.rcParams.update({'font.size': 12})
folder = "evaluation/results"
watermarks = ["Unigram", "KGW", "EXP", "SynthID", "SIR"]
all_results = {}
for watermark in tqdm(watermarks):
    results = {}
    file = f"{watermark}_final.json"
    if not os.path.exists(os.path.join(folder, file)):
        print(f"File {file} not found in {folder}")
        continue
    with open(os.path.join(folder, file), "r") as f:
        data = json.load(f)
    file = file.replace(".json", "")

    # Prepare data for scatter plot
    x = [1 - model_data["TPR"] for model_data in data.values()]
    y = [model_data["ratings"]["gpt_judge"] for model_data in data.values()]
    labels = list(data.keys())

    # Correct label mapping for all entries in the data
    final_group_styles = {
        "Paraphrase(GPT3.5)": ('o', 'blue', 'GPT3.5'),
        "Paraphrase(GPT4o)": ('s', 'blue', 'GPT4o'),
        "Paraphrase(models/Unigram_new/Qwen/Qwen2.5-3B-Instruct)": ('s', 'green', 'Ours-Qwen-3b-Unigram'),
        "Paraphrase(models/dpo_qwen_3_exponential)": ('*', 'purple', 'Ours-Qwen-3b-Exp'),
        # "none": ('X', 'black', "Original"),
    }

    # Separate labels for line plots based on suffix pattern
    suffix_line_groups = {}
    suffix_labels = {}
    for i, label in enumerate(labels):
        if "_0" in label or "_0." in label or "_1" in label:
            base_label = "_".join(label.split("_")[:-1])
            suffix = label.split("_")[-1]  # Extract the suffix value
            if base_label not in suffix_line_groups:
                suffix_line_groups[base_label] = {"x": [], "y": [], "suffixes": []}
            suffix_line_groups[base_label]["x"].append(x[i])
            suffix_line_groups[base_label]["y"].append(y[i])
            suffix_line_groups[base_label]["suffixes"].append(suffix)
    # Plot the data
    plt.figure(figsize=(12, 8))

    # Scatter plots for regular models
    for i, label in enumerate(labels):
        if "_0" in label or "_0." in label or "_1" in label:
            continue  # Skip models for line plots
        if label in final_group_styles:
            symbol, color, _label = final_group_styles[label]
            results[_label] = (x[i],  y[i])
            plt.scatter(x[i], y[i], marker=symbol, color=color,
                        label=_label if _label not in plt.gca().get_legend_handles_labels()[1] else "")

    # Line plots for suffix-based models with annotations
    for base_label, points in suffix_line_groups.items():
        sorted_points = sorted(zip(points["suffixes"], points["x"], points["y"]))  # Sort by x-values
        suffixes, x_vals, y_vals = zip(*sorted_points)
        symbol, color, _label = final_group_styles[base_label]
        plt.plot(x_vals, y_vals, label=_label, linestyle='-', marker=symbol, color=color)
        for x_val, y_val, suffix in zip(x_vals, y_vals, suffixes):
            if suffix == "0":
                res_label = _label.replace("Ours-", "").replace("-Unigram", "").replace("-Exp", "")
                results[res_label] = (x_val, y_val)
            if suffix == "1":
                results[_label] = (x_val, y_val)
            plt.text(x_val, y_val, suffix, fontsize=8, ha='center', va='bottom')  # Add suffix annotation
    all_results[watermark] = results
    # Ensure proper labels and formatting
    plt.xlabel(r"Evasion Rate $\Uparrow$")
    plt.ylabel(r"Paraphrase Quality (GPT-Judge) $\Uparrow$")
    plt.legend(title="Paraphraser", fontsize="medium", loc="best")
    plt.grid(True)

    # Save the figure with proper path handling and cleanup
    output_path = os.path.join(folder, f"{file}.pdf")
    plt.savefig(output_path, bbox_inches="tight")  # Ensure the output is tightly cropped
    plt.clf()
    plt.close()
#%%
pareto_data = {}
for watermark, results in all_results.items():
    pareto_data[watermark] = {}
    for model, (x, y) in results.items():
        is_paretto = True
        for other_model, (other_x, other_y) in results.items():
            if x < other_x and y < other_y:
                is_paretto = False
                break
        if is_paretto:
            pareto_data[watermark][model] = (x, y)
#%%
plt.figure(figsize=(12, 8))
colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
for watermark, results in pareto_data.items():
    frontier_x = [x for x, y in results.values()]
    frontier_y = [y for x, y in results.values()]
    color = next(colors)
    for model, (x, y) in results.items():
        ha = 'left' if watermark == "Unigram" else ('right' if watermark == "EXP" else 'center')
        weight = 'bold' if watermark.lower() in model.lower() else 'normal'
        plt.text(x, y, model, fontsize=8, ha=ha, va='bottom', weight=weight)
    plt.plot(frontier_x, frontier_y, label=watermark, linestyle='--', marker='o', color=color)
    gpt4o_x, gpt4o_y = all_results[watermark]["GPT4o"]
    plt.scatter(gpt4o_x, gpt4o_y, marker='s', color=color,)
    plt.text(gpt4o_x, gpt4o_y, "GPT4o", fontsize=8, ha='center', va='bottom')
plt.grid(True)
plt.xlabel(r"Evasion Rate $\Uparrow$")
plt.ylabel(r"Paraphrase Quality (GPT-Judge) $\Uparrow$")
plt.legend(title="Watermark", fontsize="medium", loc="best")
output_path = os.path.join(folder, "pareto.pdf")
plt.savefig(output_path, bbox_inches="tight")  # Ensure the output is tightly cropped
plt.clf()
plt.close()