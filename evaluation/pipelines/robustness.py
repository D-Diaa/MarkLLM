import copy
import logging
from dataclasses import dataclass
from queue import Queue
from typing import Dict, List, Any

from evaluation.dataset import BaseDataset
# Explicit imports from pipeline stages
from evaluation.pipelines.pipeline_stages import (
    GenerationStage,
    EditingStage,
    DetectionStage,
    RatingStage,
    SavingStage,
    ResponseLoader,
)
from evaluation.tools.text_editor import TextEditor
from evaluation.tools.text_quality_analyzer import TextQualityAnalyzer
from watermark.auto_watermark import AutoWatermark


@dataclass
class PipelineConfig:
    dataset: 'BaseDataset'
    edit_sequences: Dict[str, List['TextEditor']]
    rating_metrics: Dict[str, 'TextQualityAnalyzer']
    show_progress: bool = True
    batch_size: int = 4
    filename: str = "results.tsv"
    max_batch_size: int = 128


def wait_for_all(stages, result_queue):
    while True:
        try:
            item = result_queue.get()
            if item is None:
                break
        except Exception as e:
            logging.error(f"Error processing result: {str(e)}")
            break
    for stage in stages:
        stage.join()


def _feed_data(data: List[Any], queue: Queue, end=True) -> None:
    """Helper method to feed data into a queue."""
    for item in data:
        queue.put(item)
    if end:
        queue.put(None)  # Signal end of input
    logging.debug(f"Fed {len(data)} items into {queue}")


def _initialize_stages(
        stages_info: List[Dict[str, Any]],
) -> List[Any]:
    """Helper method to initialize and start pipeline stages."""
    stages = []
    for info in stages_info:
        stage = info['stage_class'](**info['init_kwargs'])
        stage.start()
        stages.append(stage)
        logging.debug(f"Started stage: {info['stage_class'].__name__}")
    return stages


class WatermarkRobustnessPipeline:
    """Pipeline for comprehensive watermark robustness evaluation"""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.total_items = self.config.dataset.prompt_nums
        self.total_generated = self.total_items * 2  # Watermarked and unwatermarked
        self.edit_expansion = len(self.config.edit_sequences)
        self.total_edited = self.total_items * self.edit_expansion + self.total_items
        logging.debug(f"Initialized pipeline with total_items={self.total_items}, "
                      f"total_generated={self.total_generated}, total_edited={self.total_edited}")

    def add_edits(self, filename: str, watermark_args: Dict[str, Any]) -> None:
        """Add edits to responses and process through the pipeline."""
        loader = ResponseLoader(filename)
        responses = loader.load_all()
        watermarked_responses = [
            r for r in responses if r.is_watermarked and r.edit_sequence == ['none']
        ]

        editing_queue = Queue()
        detection_queue = Queue()
        rating_queue = Queue()
        saving_queue = Queue()
        result_queue = Queue()

        stages_info = [
            {
                'stage_class': EditingStage,
                'init_kwargs': {
                    'input_queue': editing_queue,
                    'output_queue': detection_queue,
                    'batch_size': self.config.batch_size,
                    'total_items': len(watermarked_responses),
                    'edit_sequences': self.config.edit_sequences,
                    'show_progress': self.config.show_progress,
                },
            },
            {
                'stage_class': DetectionStage,
                'init_kwargs': {
                    'input_queue': detection_queue,
                    'output_queue': rating_queue,
                    'batch_size': min(self.edit_expansion * self.config.batch_size, self.config.max_batch_size),
                    'total_items': self.total_edited,
                    'watermark': AutoWatermark.load(**watermark_args),
                    'show_progress': self.config.show_progress,
                },
            },
            {
                'stage_class': RatingStage,
                'init_kwargs': {
                    'input_queue': rating_queue,
                    'output_queue': saving_queue,
                    'batch_size': min(self.edit_expansion * self.config.batch_size, self.config.max_batch_size),
                    'total_items': self.total_edited,
                    'rating_metrics': self.config.rating_metrics,
                    'dataset': self.config.dataset,
                    'show_progress': self.config.show_progress,
                    'use_dataset_prompts': False,
                },
            },
            {
                'stage_class': SavingStage,
                'init_kwargs': {
                    'input_queue': saving_queue,
                    'output_queue': result_queue,
                    'batch_size': min(self.edit_expansion * self.config.batch_size, self.config.max_batch_size),
                    'total_items': self.total_edited + len(responses),
                    'filename': self.config.filename,
                    'show_progress': self.config.show_progress,
                },
            },
        ]

        stages = _initialize_stages(stages_info)

        # Feed data into the pipeline
        _feed_data(watermarked_responses, editing_queue)
        _feed_data(responses, saving_queue, end=False)

        wait_for_all(stages, result_queue)

    def add_quality(self, filename: str) -> None:
        """Add quality metrics to responses and process through the pipeline."""
        loader = ResponseLoader(filename)
        responses = loader.load_all()

        rating_queue = Queue()
        saving_queue = Queue()
        result_queue = Queue()

        stages_info = [
            {
                'stage_class': RatingStage,
                'init_kwargs': {
                    'input_queue': rating_queue,
                    'output_queue': saving_queue,
                    'batch_size': self.config.batch_size,
                    'total_items': len(responses),
                    'rating_metrics': self.config.rating_metrics,
                    'dataset': self.config.dataset,
                    'show_progress': self.config.show_progress,
                    'use_dataset_prompts': False,
                },
            },
            {
                'stage_class': SavingStage,
                'init_kwargs': {
                    'input_queue': saving_queue,
                    'output_queue': result_queue,
                    'batch_size': self.config.batch_size,
                    'total_items': len(responses),
                    'filename': self.config.filename,
                    'show_progress': self.config.show_progress,
                },
            },
        ]

        stages = _initialize_stages(stages_info)

        # Feed data into the pipeline
        _feed_data(responses, rating_queue)

        wait_for_all(stages, result_queue)

    def evaluate(self, watermark_args: Dict[str, Any]) -> None:
        """Conduct full watermark robustness evaluation using the pipeline architecture."""
        generation_queue = Queue()
        editing_queue = Queue()
        detection_queue = Queue()
        rating_queue = Queue()
        saving_queue = Queue()
        result_queue = Queue()

        generation_watermark = AutoWatermark.load(**watermark_args)

        # Deepcopy tokenizer to avoid issues with multiple threads
        tokenizer_copy = copy.deepcopy(watermark_args['transformers_config'].tokenizer)
        watermark_args['transformers_config'].tokenizer = tokenizer_copy
        detection_watermark = AutoWatermark.load(**watermark_args)

        stages_info = [
            {
                'stage_class': GenerationStage,
                'init_kwargs': {
                    'input_queue': generation_queue,
                    'output_queue': editing_queue,
                    'batch_size': self.config.batch_size,
                    'total_items': self.total_items,
                    'watermark': generation_watermark,
                    'dataset': self.config.dataset,
                    'show_progress': self.config.show_progress,
                },
            },
            {
                'stage_class': EditingStage,
                'init_kwargs': {
                    'input_queue': editing_queue,
                    'output_queue': detection_queue,
                    'batch_size': min(2 * self.config.batch_size, self.config.max_batch_size),
                    'total_items': self.total_generated,
                    'edit_sequences': self.config.edit_sequences,
                    'show_progress': self.config.show_progress,
                },
            },
            {
                'stage_class': DetectionStage,
                'init_kwargs': {
                    'input_queue': detection_queue,
                    'output_queue': rating_queue,
                    'batch_size': min((self.edit_expansion + 1) * self.config.batch_size, self.config.max_batch_size),
                    'total_items': self.total_edited,
                    'watermark': detection_watermark,
                    'show_progress': self.config.show_progress,
                },
            },
            {
                'stage_class': RatingStage,
                'init_kwargs': {
                    'input_queue': rating_queue,
                    'output_queue': saving_queue,
                    'batch_size': min((self.edit_expansion + 1) * self.config.batch_size, self.config.max_batch_size),
                    'total_items': self.total_edited,
                    'rating_metrics': self.config.rating_metrics,
                    'dataset': self.config.dataset,
                    'show_progress': self.config.show_progress,
                },
            },
            {
                'stage_class': SavingStage,
                'init_kwargs': {
                    'input_queue': saving_queue,
                    'output_queue': result_queue,
                    'batch_size': min((self.edit_expansion + 1) * self.config.batch_size, self.config.max_batch_size),
                    'total_items': self.total_edited,
                    'filename': self.config.filename,
                    'show_progress': self.config.show_progress,
                },
            },
        ]

        stages = _initialize_stages(stages_info)
        _feed_data(list(range(self.total_items)), generation_queue)
        wait_for_all(stages, result_queue)
