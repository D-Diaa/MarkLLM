import copy

from evaluation.pipelines.pipeline_stages import *
from evaluation.tools.success_rate_calculator import DynamicThresholdSuccessRateCalculator
from watermark.auto_watermark import AutoWatermark


@dataclass(frozen=True)
class RobustnessEvaluationResult:
    """Complete result of robustness evaluation."""
    watermarked_responses: List[Response] = field(default_factory=list)
    unwatermarked_responses: List[Response] = field(default_factory=list)
    summary_statistics: Dict[str, Any] = field(default_factory=dict)


class WatermarkRobustnessPipeline:
    """Pipeline for comprehensive watermark robustness evaluation"""
    MAX_BATCH_SIZE = 32

    def __init__(
            self,
            dataset: BaseDataset,
            edit_sequences: Dict[str, List[TextEditor]],
            rating_metrics: Dict[str, TextQualityAnalyzer],
            show_progress: bool = True,
            batch_size: int = 4,
            filename: str = "results.tsv"
    ):
        """
        Initialize the watermark robustness evaluation pipeline.

        Args:
            dataset: Dataset containing prompts for evaluation
            edit_sequences: Dictionary mapping sequence names to lists of text editors
            rating_metrics: Dictionary mapping metric names to quality analyzers
            show_progress: Whether to show progress bars
        """
        self.dataset = dataset
        self.edit_sequences = edit_sequences
        self.rating_metrics = rating_metrics
        self.show_progress = show_progress
        self.batch_size = batch_size
        self.total_items = self.dataset.prompt_nums
        self.total_generated = self.total_items * 2  # Watermarked and unwatermarked
        self.edit_expansion = len(edit_sequences)
        self.total_edited = self.total_items * self.edit_expansion + self.total_items
        self.filename = filename

    def add_edits(self, filename: str, watermark_args: dict):
        """
        Load the pipeline from a file.
        """
        loader = ResponseLoader(filename)
        responses = loader.load_all()
        watermarked_responses = [r for r in responses if r.is_watermarked and r.edit_sequence == ['none']]
        editing_queue = Queue()  # Between generation and editing
        detection_queue = Queue()  # Between editing and detection
        rating_queue = Queue()  # Between detection and rating
        saving_queue = Queue()  # Between rating and saving
        result_queue = Queue()  # Final results
        total_generated = len(watermarked_responses)
        total_edited = total_generated * self.edit_expansion
        total_saved = total_edited + len(responses)
        # Initialize pipeline stages
        editing_stage = ParallelEditingStage(
            input_queue=editing_queue,
            output_queue=detection_queue,
            batch_size=self.batch_size,
            total_items=total_generated,
            edit_sequences=self.edit_sequences,
            show_progress=self.show_progress
        )
        detection_batch = min(self.edit_expansion * self.batch_size, self.MAX_BATCH_SIZE)
        watermark = AutoWatermark.load(**watermark_args)
        detection_stage = DetectionStage(
            input_queue=detection_queue,
            output_queue=rating_queue,
            batch_size=detection_batch,
            total_items=total_edited,
            watermark=watermark,
            show_progress=self.show_progress
        )

        rating_stage = RatingStage(
            input_queue=rating_queue,
            output_queue=saving_queue,
            batch_size=detection_batch,
            total_items=total_edited,
            rating_metrics=self.rating_metrics,
            dataset=self.dataset,
            show_progress=self.show_progress,
            use_dataset_prompts=False
        )

        saving_stage = SavingStage(
            input_queue=saving_queue,
            output_queue=result_queue,
            batch_size=detection_batch,
            total_items=total_saved,
            filename=self.filename,
            show_progress=self.show_progress
        )

        # Start all stages
        stages = [editing_stage, detection_stage, rating_stage, saving_stage]
        for stage in stages:
            stage.start()

        # Feed initial data into the pipeline
        for i in range(0, total_generated):
            editing_queue.put(watermarked_responses[i])
        editing_queue.put(None)  # Signal end of input

        # Feed responses into the saving stage
        for response in responses:
            saving_queue.put(response)

        while True:
            try:
                item = result_queue.get()
                if item is None:
                    break
                responses.append(item)
            except Exception as e:
                logging.error(f"Error processing result: {str(e)}")
                break

        # Wait for all stages to complete
        for stage in stages:
            stage.join()

        watermarked_responses = [r for r in responses if r.is_watermarked]
        unwatermarked_responses = [r for r in responses if not r.is_watermarked]

        stats = self.get_summary_stats(responses=watermarked_responses + unwatermarked_responses)

        return RobustnessEvaluationResult(
            watermarked_responses=watermarked_responses,
            unwatermarked_responses=unwatermarked_responses,
            summary_statistics=stats
        )

    def evaluate(self, watermark_args: dict) -> RobustnessEvaluationResult:
        """
        Conduct full watermark robustness evaluation using the pipeline architecture.
        """
        # Initialize queues
        generation_queue = Queue()  # Input for generation stage
        editing_queue = Queue()  # Between generation and editing
        detection_queue = Queue()  # Between editing and detection
        rating_queue = Queue()  # Between detection and rating
        saving_queue = Queue()  # Between rating and saving
        result_queue = Queue()  # Final results

        generation_watermark = AutoWatermark.load(**watermark_args)

        # Initialize pipeline stages
        generation_stage = GenerationStage(
            input_queue=generation_queue,
            output_queue=editing_queue,
            batch_size=self.batch_size,
            total_items=self.total_items,
            watermark=generation_watermark,
            dataset=self.dataset,
            show_progress=self.show_progress
        )
        editing_batch = min(2 * self.batch_size, self.MAX_BATCH_SIZE)
        editing_stage = ParallelEditingStage(
            input_queue=editing_queue,
            output_queue=detection_queue,
            batch_size=editing_batch,
            total_items=self.total_generated,
            edit_sequences=self.edit_sequences,
            show_progress=self.show_progress
        )
        # HACK: Deepcopy the tokenizer to avoid issues with multiple threads
        tokenizer_copy = copy.deepcopy(watermark_args['transformers_config'].tokenizer)
        watermark_args['transformers_config'].tokenizer = tokenizer_copy
        detection_watermark = AutoWatermark.load(**watermark_args)
        detection_batch = min((self.edit_expansion + 1) * self.batch_size, self.MAX_BATCH_SIZE)
        detection_stage = DetectionStage(
            input_queue=detection_queue,
            output_queue=rating_queue,
            batch_size=detection_batch,
            total_items=self.total_edited,
            watermark=detection_watermark,
            show_progress=self.show_progress
        )

        rating_stage = RatingStage(
            input_queue=rating_queue,
            output_queue=saving_queue,
            batch_size=detection_batch,
            total_items=self.total_edited,
            rating_metrics=self.rating_metrics,
            dataset=self.dataset,
            show_progress=self.show_progress
        )

        saving_stage = SavingStage(
            input_queue=saving_queue,
            output_queue=result_queue,
            batch_size=detection_batch,
            total_items=self.total_edited,
            filename=self.filename,
            show_progress=self.show_progress
        )

        # Start all stages
        stages = [generation_stage, editing_stage, detection_stage, rating_stage, saving_stage]
        for stage in stages:
            stage.start()

        # Feed initial data into the pipeline
        for i in range(0, self.total_items):
            generation_queue.put(i)
        generation_queue.put(None)  # Signal end of input

        # Collect results
        watermarked_responses = []
        unwatermarked_responses = []

        while True:
            try:
                item = result_queue.get()
                if item is None:
                    break

                if item.is_watermarked:
                    watermarked_responses.append(item)
                else:
                    unwatermarked_responses.append(item)

            except Exception as e:
                logging.error(f"Error processing result: {str(e)}")
                break

        # Wait for all stages to complete
        for stage in stages:
            stage.join()

        # Calculate final statistics
        stats = self.get_summary_stats(responses=watermarked_responses + unwatermarked_responses)

        return RobustnessEvaluationResult(
            watermarked_responses=watermarked_responses,
            unwatermarked_responses=unwatermarked_responses,
            summary_statistics=stats
        )

    @staticmethod
    def get_summary_stats(filename: str = None, responses: List[Response] = None) -> Dict[str, Any]:
        if responses is None:
            if filename is None:
                raise ValueError("Either filename or responses must be provided")
            loader = ResponseLoader(filename)
            responses = loader.load_all()
        watermarked_unedited = [r for r in responses if r.is_watermarked and "none" == r.edit_sequence[0]]
        unwatermarked_unedited = [r for r in responses if not r.is_watermarked and "none" == r.edit_sequence[0]]
        watermarked_unedited_scores = [r.watermark_score for r in watermarked_unedited]
        unwatermarked_unedited_scores = [r.watermark_score for r in unwatermarked_unedited]
        calculator = DynamicThresholdSuccessRateCalculator(labels=['TPR', 'F1'], rule='target_fpr', target_fpr=0.01)
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
