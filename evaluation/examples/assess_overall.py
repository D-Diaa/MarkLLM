import json
import os
from typing import Dict, List

from evaluation.dataset import MarkMyWordsDataset
from evaluation.pipelines.robustness import WatermarkRobustnessPipeline, PipelineConfig
from evaluation.tools.text_editor import TruncatePromptTextEditor, LLMParaphraser, \
    GPTParaphraser, ContextAwareSynonymSubstitution
from evaluation.tools.text_quality_analyzer import GPTTextRater
from evaluation.utils import ModelLoader, ModelConfig, _load_tokenizer
from utils.transformers_config import TransformersConfig

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
NUM_PARAPHRASES = 8
MAX_NEW_TOKENS = 256


class RobustnessPipelineFactory:
    """Factory for creating watermark robustness evaluation pipelines."""

    def __init__(self, devices: List[str]):
        self.devices = devices

    def create_pipeline(self,
                        dataset: MarkMyWordsDataset,
                        mode: str = 'end_to_end',
                        filename: str = None
                        ) -> 'WatermarkRobustnessPipeline':
        """Create a pipeline with standard configuration."""
        edit_sequences = {}
        if mode == 'gpt_judge':
            rating_metrics = {
                "gpt_judge": GPTTextRater(openai_model='gpt-4o-mini', cot=True)
            }
            batch_size = 32
        elif mode == 're_evaluate':
            rating_metrics = RobustnessPipelineFactory._create_rating_metrics(self.devices[-3:])
            batch_size = 128
        else:
            rating_metrics = RobustnessPipelineFactory._create_rating_metrics(None)
            batch_size = 16
            devices = self.devices
            if mode == 'generate_only':
                edit_sequences = {'none': [TruncatePromptTextEditor()]}
            elif mode == 'end_to_end':
                devices = devices[1:]
                edit_sequences = RobustnessPipelineFactory._create_edit_sequences(devices)
            elif mode == 'add_edits':
                edit_sequences = RobustnessPipelineFactory._create_extra_edit_sequences(devices)
            elif mode == 'base_paraphrase':
                edit_sequences = RobustnessPipelineFactory._create_base_paraphrase_sequences(devices)
            elif mode == 'ensemble_paraphrase':
                model_names = ["models/Unigram_new/Qwen/Qwen2.5-3B-Instruct", 'models/dpo_qwen_3_exponential']
                devices_per_model = len(devices) // len(model_names)
                # lambda_weights = [0, 0.2, 0.4, 0.6, 0.8, 1]
                lambda_weights = [0, 0.5, 1]
                edit_sequences = {}
                batch_size = 32
                for i, model_name in enumerate(model_names):
                    edit_sequences.update(
                        RobustnessPipelineFactory._create_weighted_paraphraser_sequences(
                            model_name, devices[i * devices_per_model:(i + 1) * devices_per_model], lambda_weights
                        )
                    )
            else:
                raise ValueError(f"Invalid mode {mode}.")
        config = PipelineConfig(dataset=dataset,
                                edit_sequences=edit_sequences,
                                rating_metrics=rating_metrics,
                                filename=filename,
                                batch_size=batch_size)
        return WatermarkRobustnessPipeline(config)

    @staticmethod
    def _create_extra_edit_sequences(devices: List[str]) -> Dict:
        """Create extra set of text editing sequences."""
        bert = ModelLoader.from_name('bert-large-uncased', device=devices[0])
        paraphraser_models = [
            # ("models/Unigram_new/Qwen/Qwen2.5-3B-Instruct", devices[0]),
            # ("models/Unigram_new/meta-llama/Llama-3.2-3B-Instruct", devices[0]),
            ("meta-llama/Llama-3.2-3B-Instruct", devices[0]),
        ]
        sequences = {
            'Word-S(Context)': [ContextAwareSynonymSubstitution(ratio=0.5, **bert)],
        }
        for model, device in paraphraser_models:
            paraphraser = RobustnessPipelineFactory._create_paraphraser(model, device)
            # no prompt truncate, because "none" is already truncated
            sequences[f'Paraphrase({model})'] = [paraphraser]
        return sequences

    @staticmethod
    def _create_edit_sequences(devices: List[str]) -> Dict:
        """Create standard set of text editing sequences."""
        base_editor = TruncatePromptTextEditor()
        sequences = {
            'none': [base_editor],
            # 'Word-D': [base_editor, WordDeletion(ratio=0.3)],
            # 'Word-S': [base_editor, SynonymSubstitution(ratio=0.5)],
            'Paraphrase(GPT3.5)': [GPTParaphraser('gpt-3.5-turbo')],
            'Paraphrase(GPT4o)': [GPTParaphraser('gpt-4o')]
        }

        # Add paraphrasers
        # paraphraser_models = [
        #     ("models/Unigram_new/Qwen/Qwen2.5-3B-Instruct", devices[1]),
        #     ("models/Unigram_new/meta-llama/Llama-3.2-3B-Instruct", devices[3]),
        #     ("models/Unigram/Qwen/Qwen2.5-3B-Instruct", devices[2]),
        #     ("models/Unigram/meta-llama/Llama-3.2-3B-Instruct", devices[2]),
        #     ("Qwen/Qwen2.5-3B-Instruct", devices[1]),
        #     ("models/dpo_qwen_3_exponential", devices[2]),
        #     ('meta-llama/Llama-3.1-8B-Instruct', devices[4]),
        #     ("models/dpo_llama_3_all", devices[2])
        # ]
        #
        # for model, device in paraphraser_models:
        #     paraphraser = RobustnessPipelineFactory._create_paraphraser(model, device)
        #     sequences[f'Paraphrase({model})'] = [base_editor, paraphraser]

        model_names = ["models/Unigram_new/Qwen/Qwen2.5-3B-Instruct", 'models/dpo_qwen_3_exponential']
        devices_per_model = len(devices) // len(model_names)
        # lambda_weights = [0, 0.2, 0.4, 0.6, 0.8, 1]
        lambda_weights = [0, 0.2, 0.4, 0.6, 0.8, 1]
        for i, model_name in enumerate(model_names):
            sequences.update(
                RobustnessPipelineFactory._create_weighted_paraphraser_sequences(
                    model_name, devices[i * devices_per_model:(i + 1) * devices_per_model], lambda_weights
                )
            )

        return sequences

    @staticmethod
    def _create_base_paraphrase_sequences(devices: List[str]) -> Dict:
        """Create base set of text editing sequences."""
        paraphraser_models = [
            ("Qwen/Qwen2.5-3B-Instruct", devices[0]),
            ('meta-llama/Llama-3.2-3B-Instruct', devices[0]),
        ]
        sequences = {
        }
        for model, device in paraphraser_models:
            paraphraser = RobustnessPipelineFactory._create_paraphraser(model, device,
                                                                        num_return_sequences=NUM_PARAPHRASES)
            sequences[f'Paraphrase({model})'] = [paraphraser]
        return sequences

    @staticmethod
    def _create_paraphraser(model_name: str, device: str, **kwargs) -> LLMParaphraser:
        """Create a paraphraser instance."""

        return LLMParaphraser(
            model_config=ModelConfig(model_name=model_name, device=device),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=1.0,
            **kwargs
        )

    @staticmethod
    def _create_weighted_paraphraser(model_name: str, device: str, weight: float, **kwargs) -> LLMParaphraser:
        """Create a paraphraser instance."""
        return LLMParaphraser(
            model_config=ModelConfig(model_name=model_name, device=device),
            adapter_ratio=weight,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=1.0,
            **kwargs
        )

    @staticmethod
    def _create_weighted_paraphraser_sequences(model_name: str, devices: List[str], weights: List[int], **kwargs):
        sequences = {}
        for i, weight in enumerate(weights):
            idx = i % len(devices)
            sequences[f'Paraphrase({model_name})_{weight}'] = [
                RobustnessPipelineFactory._create_weighted_paraphraser(
                    model_name, devices[idx], weight, **kwargs
                )
            ]
        return sequences

    @staticmethod
    def _create_ensemble_paraphrasers(model_name: str, devices: List[str], lambda_weights: List[int], **kwargs):
        model_dicts = [ModelLoader.from_name(model_name, device=device) for device in devices]
        base_model_name = model_dicts[0]['model'].config.name_or_path
        base_model_dicts = [ModelLoader.from_name(base_model_name, device=device) for device in devices]
        paraphrasers = {}
        for i, lambda_weight in enumerate(lambda_weights):
            idx = i % len(devices)
            paraphrasers[f'Paraphrase({model_name})_{lambda_weight}'] = [
                RobustnessPipelineFactory._create_ensemble_paraphraser(
                    model_dicts[idx], base_model_dicts[idx], devices[idx], lambda_weight, **kwargs
                )
            ]

        return paraphrasers

    @staticmethod
    def _create_rating_metrics(devices: List[str]) -> Dict:
        """Create standard set of rating metrics."""
        # rating_models = [ModelLoader.from_name('meta-llama/Llama-3.1-8B-Instruct', device=device) for device in devices]

        return {
            # "llm_judge": LLMTextRater(
            #     **rating_models[0],
            #     device=devices[0],
            #     cot=False,
            #     max_new_tokens=MAX_NEW_TOKENS,
            #     do_sample=False,
            #     top_p=None,
            #     temperature=None
            # ),
            # "ppl": PPLCalculator(**rating_models[1], device=devices[1]),
            # "llm_cot": LLMTextRater(
            #     **rating_models[2],
            #     device=devices[2],
            #     cot=True,
            #     max_new_tokens=1024,
            #     do_sample=False,
            #     top_p=None,
            #     temperature=None
            # ),
            "gpt_judge": GPTTextRater(openai_model='gpt-4o-mini', cot=True)
        }


# main.py
def main():
    import argparse
    from torch.cuda import device_count

    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='KGW')
    parser.add_argument(
        '--mode',
        type=str,
        default='end_to_end',
        choices=['end_to_end', 'generate_only', 'add_edits', 'base_paraphrase', 'gpt_judge', 'ensemble_paraphrase',
                 're_evaluate']
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='eval',
        choices=['eval', 'train']
    )
    parser.add_argument('--suff', type=str, required=False, default="_finaleval")
    parser.add_argument('--samples', type=int, default=296)
    parser.add_argument(
        '--hash_keys',
        type=int,  # Specifies that each input should be converted to an integer
        nargs='+',  # Allows one or more integers (use '*' for zero or more)
        help="A list of hash_keys to use for watermarking",
        default=[15485863]
    )
    directory = "evaluation/results"
    args = parser.parse_args()

    # Setup devices
    devices = [f'cuda:{i}' for i in range(device_count())]
    pipeline_factory = RobustnessPipelineFactory(devices)
    # Initialize model and dataset
    provider_config = ModelConfig(
        model_name='meta-llama/Llama-3.1-8B-Instruct',
        device=devices[0]
    )
    if args.mode in ['generate_only', 'end_to_end']:
        provider_model, provider_tokenizer = ModelLoader(provider_config).load()
    else:
        provider_model, provider_tokenizer = None, _load_tokenizer(provider_config.model_name)
    dataset = MarkMyWordsDataset(provider_tokenizer, max_samples=args.samples, dataset=args.dataset)
    watermark_config_file = f'config/{args.algorithm}.json'
    with open(watermark_config_file, 'r') as f:
        config = json.load(f)
    hash_keys = args.hash_keys
    if "hash_key" not in config:
        hash_keys = [None]  # Just use the default hash key for unsupported algorithms
    for hash_key in hash_keys:
        # change the hash key in the config file
        if "hash_key" in config:
            config['hash_key'] = hash_key
        with open(watermark_config_file, 'w') as f:
            json.dump(config, f)

        watermark_args = {
            'algorithm_name': args.algorithm,
            'algorithm_config': watermark_config_file,
            'transformers_config': TransformersConfig(
                model=provider_model,
                tokenizer=provider_tokenizer,
                vocab_size=len(provider_tokenizer),
                device=provider_config.device,
                max_new_tokens=provider_config.max_new_tokens,
                do_sample=provider_config.do_sample
            )
        }
        if args.suff:
            suff = args.suff
        else:
            suff = "_gen" if args.mode == 'generate_only' else "_edit"
        # Create and run pipeline based on mode
        pipeline = pipeline_factory.create_pipeline(
            dataset=dataset,
            mode=args.mode,
            filename=f'{directory}/{args.algorithm}{suff}{hash_key}.tsv'
        )
        if args.mode in ['generate_only', 'end_to_end']:
            pipeline.evaluate(watermark_args)
        elif args.mode in ['gpt_judge', 're_evaluate']:
            pipeline.add_quality(f'{directory}/{args.algorithm}_new{hash_key}.tsv')
        else:
            if args.mode == 'base_paraphrase':
                pipeline.edit_expansion = len(pipeline.config.edit_sequences) * NUM_PARAPHRASES
            pipeline.add_edits(f'{directory}/{args.algorithm}_finaleval{hash_key}.tsv', watermark_args)


if __name__ == '__main__':
    main()
