import json
import os
from typing import Dict, Any

from tqdm import tqdm

from evaluation.pipelines.pipeline_stages import ResponseLoader
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator


def get_summary_stats(filename: str = None, reverse=None) -> Dict[str, Any]:
    loader = ResponseLoader(filename)
    responses = loader.load_all()
    watermarked_unedited = [r for r in responses if r.is_watermarked and "none" == r.edit_sequence[0]]
    unwatermarked_unedited = [r for r in responses if not r.is_watermarked and "none" == r.edit_sequence[0]]
    if reverse is None and filename is not None:
        reverse = "EXP" in filename
    watermarked_unedited_scores = [r.watermark_score for r in watermarked_unedited]
    unwatermarked_unedited_scores = [r.watermark_score for r in unwatermarked_unedited]
    calculator = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'F1'], rule='target_fpr', target_fpr=0.01,
                                                       reverse=reverse)
    threshold = calculator.find_threshold(watermarked_unedited_scores, unwatermarked_unedited_scores)
    grouped_by_sequence = {}
    for response in responses:
        if not response.is_watermarked:
            continue
        sequence = response.edit_sequence[0]
        if sequence not in grouped_by_sequence:
            grouped_by_sequence[sequence] = []
        grouped_by_sequence[sequence].append(response)
    stats_per_group = {}
    rating_keys = responses[0].ratings.keys()
    for sequence, group in grouped_by_sequence.items():
        watermark_scores = [r.watermark_score for r in group]
        stats_per_group[sequence] = calculator.calculate_from_threshold(watermark_scores, threshold)
        stats_per_group[sequence]["default_detection"] = sum(r.watermark_detected for r in group) / len(group)
        stats_per_group[sequence]["watermark_scores"] = sum(r.watermark_score for r in group) / len(group)
        stats_per_group[sequence]["ratings"] = {
            rating_key: sum(r.ratings[rating_key] for r in group) / len(group) for rating_key in rating_keys
        }

    return stats_per_group


folder = "evaluation/results"
files = tqdm(list(os.listdir(folder)))
for file in files:
    if file.endswith(".tsv") and "ensemble" in file:
        file_path = os.path.join(folder, file)
        stats = get_summary_stats(filename=file_path)
        # remove tsv
        file_name = file.replace(".tsv", "") + ".json"
        with open(os.path.join(folder, file_name), "w") as f:
            json.dump(stats, f)
