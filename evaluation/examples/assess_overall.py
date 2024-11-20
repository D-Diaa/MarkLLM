import json
import os
from dataclasses import dataclass
from typing import Dict, List
from typing import Tuple

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BertTokenizer, BertForMaskedLM

from evaluation.dataset import MarkMyWordsDataset
from evaluation.pipelines.robustness import WatermarkRobustnessPipeline
from evaluation.tools.text_editor import TruncatePromptTextEditor, WordDeletion, SynonymSubstitution, LLMParaphraser, \
    GPTParaphraser, ContextAwareSynonymSubstitution
from evaluation.tools.text_quality_analyzer import LLMTextRater, PPLCalculator, GPTTextRater
from utils.transformers_config import TransformersConfig

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
NUM_PARAPHRASES = 8


@dataclass
class ModelConfig:
    """Configuration for model loading and inference."""
    model_name: str
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_new_tokens: int = 256
    do_sample: bool = True


def _load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load tokenizer with error handling."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        tokenizer.padding_side = "left"
        return tokenizer
    except Exception as e:
        raise ValueError(f"Failed to load tokenizer {model_name}.") from e


class ModelLoader:
    """Handles loading and configuration of language models."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model and tokenizer based on configuration."""
        if self._is_adapter_model():
            print("Loading model with PEFT adapter.")
            self._load_adapter_model()
        else:
            print("Loading base model.")
            self._load_base_model()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        return self.model, self.tokenizer

    def _is_adapter_model(self) -> bool:
        """Check if the model path contains an adapter configuration."""
        model_path = self.config.model_name
        adapter_config_path = os.path.join(model_path, 'adapter_config.json')
        return (os.path.exists(model_path) and
                not os.path.exists(os.path.join(model_path, 'config.json')) and
                os.path.exists(adapter_config_path))

    def _load_adapter_model(self):
        """Load a model with PEFT adapter."""
        peft_config = PeftConfig.from_pretrained(self.config.model_name)
        base_model = self._load_model(peft_config.base_model_name_or_path)
        self.model = PeftModel.from_pretrained(base_model, self.config.model_name)
        self.tokenizer = _load_tokenizer(peft_config.base_model_name_or_path)

    def _load_base_model(self):
        """Load a base model without adapters."""
        self.model = self._load_model(self.config.model_name)
        self.tokenizer = _load_tokenizer(self.config.model_name)

    def _load_model(self, model_name: str) -> AutoModelForCausalLM:
        """Load model with error handling."""
        try:
            return AutoModelForCausalLM.from_pretrained(model_name).to(self.config.device)
        except Exception as e:
            raise ValueError(f"Failed to load model {model_name}.") from e

    @classmethod
    def from_name(cls, model_name: str, device: str = None) -> Dict:
        """Create a model loader from model name and optional device."""
        config = ModelConfig(model_name=model_name)
        if device:
            config.device = device
        model, tokenizer = cls(config).load()
        return {'model': model, 'tokenizer': tokenizer}


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
        if mode == 'gpt_judge':
            rating_metrics = {
                "gpt_judge": GPTTextRater(openai_model='gpt-4o-mini', cot=True)
            }
            edit_sequences = {}
            batch_size = 32
        else:
            rating_metrics = RobustnessPipelineFactory._create_rating_metrics(self.devices[-1])
            if mode == 'generate_only':
                edit_sequences = {'none': [TruncatePromptTextEditor()]}
            elif mode == 'end_to_end':
                edit_sequences = RobustnessPipelineFactory._create_edit_sequences(self.devices[:-1])
            elif mode == 'add_edits':
                edit_sequences = RobustnessPipelineFactory._create_extra_edit_sequences(self.devices[:-1])
            elif mode == 'base_paraphrase':
                edit_sequences = RobustnessPipelineFactory._create_base_paraphrase_sequences(self.devices[:-1])
            else:
                raise ValueError(f"Invalid mode {mode}.")
            batch_size = 16

        return WatermarkRobustnessPipeline(
            dataset=dataset,
            edit_sequences=edit_sequences,
            rating_metrics=rating_metrics,
            filename=filename,
            batch_size=batch_size
        )

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
            'Word-D': [base_editor, WordDeletion(ratio=0.3)],
            'Word-S': [base_editor, SynonymSubstitution(ratio=0.5)],
            'Paraphrase(GPT3.5)': [GPTParaphraser('gpt-3.5-turbo')],
            'Paraphrase(GPT4o)': [GPTParaphraser('gpt-4o')]
        }

        # Add paraphrasers
        paraphraser_models = [
            ("models/Unigram_new/Qwen/Qwen2.5-3B-Instruct", devices[3]),
            ("models/Unigram_new/meta-llama/Llama-3.2-3B-Instruct", devices[3]),
            # ("models/Unigram/Qwen/Qwen2.5-3B-Instruct", devices[2]),
            # ("models/Unigram/meta-llama/Llama-3.2-3B-Instruct", devices[2]),
            ("Qwen/Qwen2.5-3B-Instruct", devices[1]),
            ("models/dpo_qwen_3_exponential", devices[1]),
            # ('meta-llama/Llama-3.1-8B-Instruct', devices[4]),
            # ("models/dpo_llama_3_all", devices[2])
        ]

        for model, device in paraphraser_models:
            paraphraser = RobustnessPipelineFactory._create_paraphraser(model, device)
            sequences[f'Paraphrase({model})'] = [base_editor, paraphraser]

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
        model_dict = ModelLoader.from_name(model_name, device=device)
        return LLMParaphraser(
            **model_dict,
            device=device,
            max_new_tokens=256,
            do_sample=True,
            temperature=1.0,
            **kwargs
        )

    @staticmethod
    def _create_rating_metrics(device: str) -> Dict:
        """Create standard set of rating metrics."""
        rating_model = ModelLoader.from_name('meta-llama/Llama-3.1-8B-Instruct', device=device)
        return {
            "llm_judge": LLMTextRater(
                **rating_model,
                device=device,
                cot=False,
                max_new_tokens=32,
                do_sample=False,
                top_p=None,
                temperature=None
            ),
            "ppl": PPLCalculator(**rating_model, device=device)
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
        choices=['end_to_end', 'generate_only', 'add_edits', 'base_paraphrase', 'gpt_judge']
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
            stats = pipeline.evaluate(watermark_args)
        elif args.mode == "gpt_judge":
            stats = pipeline.add_quality(f'{directory}/{args.algorithm}_final{hash_key}.tsv')
        else:
            if args.mode == 'base_paraphrase':
                pipeline.edit_expansion = len(pipeline.edit_sequences) * NUM_PARAPHRASES
            stats = pipeline.add_edits(f'{directory}/{args.algorithm}_gen{hash_key}.tsv', watermark_args)

        # Save results
        os.makedirs('evaluation/results', exist_ok=True)
        with open(f'{directory}/{args.algorithm}{suff}{hash_key}.json', 'w') as f:
            json.dump(stats.summary_statistics, f)


if __name__ == '__main__':
    main()
