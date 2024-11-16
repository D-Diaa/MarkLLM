import random
from collections import defaultdict
from typing import List

import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from evaluation.pipelines.pipeline_stages import ResponseLoader

system_prompt = (
    "You are an expert copy-editor. Please rewrite the following text in your own voice and paraphrase all sentences.\n"
    "Ensure that the final output contains the same information as the original text and has roughly the same length.\n"
    "Do not leave out any important details when rewriting in your own voice. Do not include any information that is not"
    "present in the original text. Do not respond with a greeting or any other extraneous information. "
    "Skip the preamble. Just rewrite the text directly."
)
instruction = "\n[[START OF TEXT]]\n{}\n[[END OF TEXT]]"
response = "[[START OF PARAPHRASE]]\n"


def create_dpo_dataset(input_files: List[str], output_dir: str, model: str) -> Dataset:

    tokenizer = AutoTokenizer.from_pretrained(model)
    dpo_data = []
    rate_thresh = 0.8
    for input_file in input_files:
        loader = ResponseLoader(input_file)
        generations = loader.load_all()

        watermarked_generations = [g for g in generations if g.is_watermarked]
        # unwatermarked_generations = [g for g in generations if not g.is_watermarked]
        # calculator = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'F1'], rule='target_fpr', target_fpr=0.05)
        # watermarked_unattacked_scores = [g.watermark_score for g in watermarked_generations if
        #                                  "none" in str(g.edit_sequence)]
        # unwatermarked_scores = [g.watermark_score for g in unwatermarked_generations]
        # wm_thresh = calculator.find_threshold(watermarked_unattacked_scores, unwatermarked_scores)
        watermark_group = defaultdict(list)
        for gen in watermarked_generations:
            watermark_group[gen.id].append(gen)

        for _id in tqdm(watermark_group):
            id_group = watermark_group[_id]
            edit_strs = [str(g.edit_sequence) for g in id_group]
            no_attack = [g for g, edit in zip(id_group, edit_strs) if "none" in edit]
            paraphrases = [g for g, edit in zip(id_group, edit_strs) if model in edit]
            if not no_attack or not paraphrases:
                continue

            if not any(no_attack[i].watermark_detected for i in range(len(no_attack))):
                continue
            # no_attack_rating = no_attack[0].ratings["llm_judge"]
            # rate_thresh = min(0.8, no_attack_rating)
            chosen_paraphrases = [g for g in paraphrases if g.ratings["llm_judge"] > rate_thresh and not g.watermark_detected]
            rejected_paraphrases = [g for g in paraphrases if g.ratings["llm_judge"] < rate_thresh or g.watermark_detected]
            if len(chosen_paraphrases) == 0:
                continue
            if len(rejected_paraphrases) == 0:
                rejected_paraphrases = no_attack

            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction.format(no_attack[0].text)}
                ],
                tokenize=False,
                add_generation_prompt=True
            ) + response
            for i, p in enumerate(chosen_paraphrases):
                dpo_data.append({
                    'prompt': prompt,
                    'chosen': p.text,
                    'rejected': random.choice(rejected_paraphrases).text,
                })
    dataset = Dataset.from_pandas(pd.DataFrame(dpo_data))
    dataset.save_to_disk(output_dir)
    print(f"Created DPO dataset with {len(dpo_data)} samples")
    return dataset


if __name__ == "__main__":
    watermark = "Unigram"
    keys = list(range(10, 330, 10))
    input_files = [
        f"evaluation/results/{watermark}_edit{key}.tsv" for key in keys
    ]
    model = 'meta-llama/Llama-3.2-3B-Instruct'
    # model = "Qwen/Qwen2.5-3B-Instruct"
    output_dir = f"data/{watermark}_new/{model}"
    create_dpo_dataset(input_files, output_dir, model)
