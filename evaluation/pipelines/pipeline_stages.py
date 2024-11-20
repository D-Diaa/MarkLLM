import ast
import csv
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from dataclasses import replace, field
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any
from typing import List, Dict, Iterator, Optional

from tqdm import tqdm

from evaluation.dataset import BaseDataset
from evaluation.tools.text_editor import TextEditor, TruncatePromptTextEditor
from evaluation.tools.text_quality_analyzer import TextQualityAnalyzer, LLMTextRater, ReferencedTextQualityAnalyzer, \
    DirectTextQualityAnalyzer, GPTTextRater
from watermark.base import BaseWatermark


@dataclass(frozen=True)
class Response:
    """Class to hold response data and evaluation metrics"""
    id: int
    text: str
    prompt: str
    watermark_detected: bool = False
    is_watermarked: bool = False
    edit_sequence: List[str] = field(default_factory=list)
    watermark_score: float = None
    ratings: Dict[str, float] = field(default_factory=dict)


class PipelineStage(Thread):
    """Base class for pipeline stages"""

    def __init__(self, input_queue: Queue, output_queue: Queue,
                 batch_size: int, total_items: int, stage_name: str,
                 show_progress: bool = True):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.batch_size = batch_size
        self.total_items = total_items
        self.stage_name = stage_name
        self.show_progress = show_progress
        self.progress = tqdm(total=total_items,
                             desc=f"{stage_name}",
                             disable=not show_progress)

    def process_batch(self, batch: List[Any]) -> List[Any]:
        """Override this method to implement batch processing logic"""
        raise NotImplementedError

    def run(self):
        current_batch = []
        try:
            while True:
                item = self.input_queue.get()
                if item is None:
                    # Process remaining items in the last batch
                    if current_batch:
                        processed = self.process_batch(current_batch)
                        for item in processed:
                            # One to many output
                            if isinstance(item, list):
                                for ii in item:
                                    self.output_queue.put(ii)
                            else:
                                self.output_queue.put(item)
                        self.progress.update(len(current_batch))
                    break

                current_batch.append(item)
                if len(current_batch) >= self.batch_size:
                    processed = self.process_batch(current_batch)
                    for item in processed:
                        self.output_queue.put(item)
                    self.progress.update(len(current_batch))
                    current_batch = []

            # Signal next stage
            self.output_queue.put(None)
        except Exception as e:
            logging.error(f"Error in {self.stage_name}: {str(e)}")
            # Ensure pipeline termination on error
            self.output_queue.put(None)
        finally:
            self.progress.close()


class GenerationStage(PipelineStage):
    """Stage for generating watermarked and unwatermarked text"""

    def __init__(self, input_queue: Queue, output_queue: Queue,
                 batch_size: int, total_items: int, watermark: BaseWatermark,
                 dataset: BaseDataset, show_progress: bool = True):
        super().__init__(input_queue, output_queue, batch_size, total_items,
                         "Generation", show_progress)
        self.watermark = watermark
        self.dataset = dataset

    def process_batch(self, batch: List[int]) -> List[Response]:
        prompts = [self.dataset.get_prompt(idx) for idx in batch]
        watermarked_texts = self.watermark.generate_watermarked_texts(prompts)
        unwatermarked_texts = self.watermark.generate_unwatermarked_texts(prompts)
        responses = [
            Response(id=idx, text=watermarked_text, prompt=prompt, is_watermarked=True)
            for idx, prompt, watermarked_text in zip(batch, prompts, watermarked_texts)
        ]
        responses.extend([
            Response(id=idx, text=unwatermarked_text, prompt=prompt, is_watermarked=False)
            for idx, prompt, unwatermarked_text in zip(batch, prompts, unwatermarked_texts)
        ])
        return responses


class EditingStage(PipelineStage):
    """Stage for applying edit sequences to texts"""

    def __init__(self, input_queue: Queue, output_queue: Queue,
                 batch_size: int, total_items: int,
                 edit_sequences: Dict[str, List[TextEditor]],
                 show_progress: bool = True):
        super().__init__(input_queue, output_queue, batch_size, total_items,
                         "Editing", show_progress)
        self.edit_sequences = edit_sequences
        self.non_watermarked_edit_sequence = [TruncatePromptTextEditor()]

    def _apply_edit_sequence_batched(
            self,
            texts: List[str],
            prompts: List[str],
            editors: List[TextEditor],
    ) -> List[str]:
        """Apply a sequence of edits to text and return Response object"""
        edited_texts = texts
        for editor in editors:
            edited_texts = editor.edit_batch(edited_texts, prompts)

        return edited_texts

    def edit_batch(self, batch: List[Response], editors: List[TextEditor], seq_name: str) -> List[Response]:
        edited_responses = []
        edited_batch = self._apply_edit_sequence_batched(
            [r.text for r in batch],
            [r.prompt for r in batch],
            editors,
        )
        for i, edited_text in enumerate(edited_batch):
            if isinstance(edited_text, list):
                # Handle one-to-many edits
                for et in edited_text:
                    response = replace(batch[i], text=et, edit_sequence=[seq_name])
                    edited_responses.append(response)
            else:
                response = replace(batch[i], text=edited_text, edit_sequence=[seq_name])
                edited_responses.append(response)
        return edited_responses

    def apply_edits(
            self,
            responses: List[Response],
            apply_non_watermarked: bool = False
    ) -> List[Response]:
        """Apply all edit sequences to the responses"""
        edited_responses = []
        edit_sequences = self.edit_sequences if not apply_non_watermarked else {
            'none': self.non_watermarked_edit_sequence
        }
        # Apply each edit sequence
        batched_responses = [responses[i:i + self.batch_size] for i in range(0, len(responses), self.batch_size)]
        for seq_name, editors in edit_sequences.items():
            for batch in batched_responses:
                edited_batch = self.edit_batch(batch, editors, seq_name)
                edited_responses.extend(edited_batch)
        return edited_responses

    def process_batch(self, batch: List[Response]) -> List[Response]:
        watermarked_batch = [r for r in batch if r.is_watermarked]
        unwatermarked_batch = [r for r in batch if not r.is_watermarked]
        edited_watermarked = self.apply_edits(watermarked_batch)
        edited_unwatermarked = self.apply_edits(unwatermarked_batch, apply_non_watermarked=True)
        return edited_watermarked + edited_unwatermarked


class ParallelEditingStage(EditingStage):
    """Stage for applying edit sequences to texts in parallel using threads"""

    def __init__(self, input_queue: Queue, output_queue: Queue,
                 batch_size: int, total_items: int,
                 edit_sequences: Dict[str, List[TextEditor]],
                 max_workers: int = None,
                 show_progress: bool = True):
        super().__init__(input_queue, output_queue, batch_size, total_items,
                         edit_sequences, show_progress)
        # Count number of paraphrase sequences for default max_workers
        paraphrase_count = sum(1 for name in edit_sequences.keys() if 'Paraphrase' in name)
        self.max_workers = max_workers or paraphrase_count

    def process_edit_sequence(self, responses: List[Response],
                              seq_name: str,
                              editors: List[TextEditor]) -> List[Response]:
        """Process a single edit sequence for all responses"""
        edited_batch = self._apply_edit_sequence_batched(
            [r.text for r in responses],
            [r.prompt for r in responses],
            editors
        )
        edited_responses = []
        for i, edited_text in enumerate(edited_batch):
            if isinstance(edited_text, list):
                # Handle one-to-many edits
                for et in edited_text:
                    response = replace(responses[i], text=et, edit_sequence=[seq_name])
                    edited_responses.append(response)
            else:
                response = replace(responses[i], text=edited_text, edit_sequence=[seq_name])
                edited_responses.append(response)
        return edited_responses

    def apply_edits(
            self,
            responses: List[Response],
            apply_non_watermarked: bool = False
    ) -> List[Response]:
        """Apply all edit sequences to the responses, with parallel processing for Paraphrase sequences"""
        edit_sequences = self.edit_sequences if not apply_non_watermarked else {
            'none': self.non_watermarked_edit_sequence
        }

        edited_responses = []

        # Separate paraphrase and non-paraphrase sequences
        paraphrase_sequences = {
            name: editors for name, editors in edit_sequences.items()
            if 'Paraphrase' in name
        }
        other_sequences = {
            name: editors for name, editors in edit_sequences.items()
            if 'Paraphrase' not in name
        }

        # Process paraphrase sequences in parallel
        if paraphrase_sequences:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(
                        self.process_edit_sequence,
                        responses,
                        seq_name,
                        editors
                    )
                    for seq_name, editors in paraphrase_sequences.items()
                ]

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        edited_responses.extend(result)
                    except Exception as e:
                        logging.error(f"Error in paraphrase sequence processing: {str(e)}")
                        continue

        # Process other sequences sequentially in the main thread
        for seq_name, editors in other_sequences.items():
            try:
                result = self.process_edit_sequence(responses, seq_name, editors)
                edited_responses.extend(result)
            except Exception as e:
                logging.error(f"Error in sequential sequence processing: {str(e)}")
                continue

        return edited_responses

    def process_batch(self, batch: List[Response]) -> List[Response]:
        """Process a batch of responses, splitting watermarked and unwatermarked texts"""
        watermarked_batch = [r for r in batch if r.is_watermarked]
        unwatermarked_batch = [r for r in batch if not r.is_watermarked]

        edited_watermarked = self.apply_edits(watermarked_batch) if watermarked_batch else []
        edited_unwatermarked = self.apply_edits(unwatermarked_batch,
                                                apply_non_watermarked=True) if unwatermarked_batch else []

        return edited_watermarked + edited_unwatermarked


class DetectionStage(PipelineStage):
    """Stage for detecting watermarks in texts"""

    def __init__(self, input_queue: Queue, output_queue: Queue,
                 batch_size: int, total_items: int, watermark: BaseWatermark,
                 show_progress: bool = True):
        super().__init__(input_queue, output_queue, batch_size, total_items,
                         "Detection", show_progress)
        self.watermark = watermark

    def process_batch(self, batch: List[Response]) -> List[Response]:
        detected_responses = []
        for response in batch:
            result = self.watermark.detect_watermark(response.text, return_dict=True)
            detected_response = replace(response,
                                        watermark_detected=result['is_watermarked'],
                                        watermark_score=result['score'])
            detected_responses.append(detected_response)
        return detected_responses


class RatingStage(PipelineStage):
    """Stage for computing quality ratings"""

    def __init__(self, input_queue: Queue, output_queue: Queue,
                 batch_size: int, total_items: int,
                 rating_metrics: Dict[str, TextQualityAnalyzer], dataset: BaseDataset,
                 show_progress: bool = True, use_dataset_prompts: bool = True):
        super().__init__(input_queue, output_queue, batch_size, total_items,
                         "Rating", show_progress)
        self.rating_metrics = rating_metrics
        self.dataset = dataset
        self.use_dataset_prompts = use_dataset_prompts

    def compute_ratings(
            self,
            responses: List[Response]
    ) -> List[Response]:
        """Compute quality ratings for all responses"""
        batched_responses = [responses[i:i + self.batch_size] for i in range(0, len(responses), self.batch_size)]
        rated_responses = []
        for batch in batched_responses:
            ratings = {}
            for metric_name, analyzer in self.rating_metrics.items():
                # Handle different analyzer types
                if isinstance(analyzer, LLMTextRater) or isinstance(analyzer, GPTTextRater):
                    if hasattr(self.dataset, 'raw_prompts') and self.use_dataset_prompts:
                        prompts = [self.dataset.raw_prompts[r.id][0] for r in batch]
                        batch = [replace(r, prompt=p) for r, p in zip(batch, prompts)]
                    prompts = [r.prompt for r in batch]
                    scores = analyzer.analyze_batched([r.text for r in batch], prompts)
                elif isinstance(analyzer, ReferencedTextQualityAnalyzer):
                    # For referenced analyzers
                    references = [self.dataset.get_reference(r.id) for r in batch]
                    scores = analyzer.analyze_batched([r.text for r in batch], references)
                elif isinstance(analyzer, DirectTextQualityAnalyzer):
                    # For direct analyzers
                    scores = analyzer.analyze_batched([r.text for r in batch])
                else:
                    raise ValueError(f"Unsupported analyzer type: {type(analyzer)}")
                ratings[metric_name] = scores
            for i, response in enumerate(batch):
                sample_ratings = {metric_name: scores[i] for metric_name, scores in ratings.items()}
                sample_ratings.update(response.ratings)
                rated_response = replace(response, ratings=sample_ratings)
                rated_responses.append(rated_response)
        return rated_responses

    def process_batch(self, batch: List[Response]) -> List[Response]:
        rated_responses = self.compute_ratings(batch)
        return rated_responses


class SavingStage(PipelineStage):
    """Stage for saving responses to a CSV file."""

    def __init__(self, input_queue: Queue, output_queue: Queue,
                 batch_size: int, total_items: int, filename: str,
                 show_progress: bool = True):
        super().__init__(input_queue, output_queue, batch_size, total_items,
                         "Saving", show_progress)
        self.filename = filename
        self.fieldnames = [
            "id", "text", "prompt", "watermark_detected",
            "is_watermarked", "edit_sequence", "watermark_score",
            "ratings"
        ]

        self._ensure_directory_exists()
        self._initialize_file()

    def _ensure_directory_exists(self):
        """Ensure that the directory for the file exists."""
        directory = os.path.dirname(os.path.abspath(self.filename))
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                logging.info(f"Created directory: {directory}")
            except OSError as e:
                logging.error(f"Failed to create directory {directory}: {e}")
                raise

    def _initialize_file(self):
        """Initialize the CSV file with headers if it does not exist."""
        file_exists = os.path.isfile(self.filename)
        try:
            self.file = open(self.filename, 'a', newline='', encoding='utf-8')
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames, delimiter='\t')
            if not file_exists:
                self.writer.writeheader()
                logging.info(f"Initialized new CSV file with headers: {self.filename}")
        except OSError as e:
            logging.error(f"Failed to open file {self.filename}: {e}")
            raise

    def process_batch(self, batch: List['Response']) -> List['Response']:
        """Process a batch of responses by writing them to the CSV file."""
        try:
            for response in batch:
                row = self._response_to_dict(response)
                self.writer.writerow(row)
            self.file.flush()
            logging.debug(f"Successfully wrote batch of {len(batch)} responses to {self.filename}")
        except Exception as e:
            logging.error(f"Error writing batch to file {self.filename}: {e}")
            # Depending on requirements, you might want to handle the exception differently
        return batch

    def _response_to_dict(self, response: 'Response') -> dict:
        """Convert a Response object to a dictionary suitable for CSV writing."""
        return {
            "id": response.id,
            "text": response.text,
            "prompt": response.prompt,
            "watermark_detected": response.watermark_detected,
            "is_watermarked": response.is_watermarked,
            "edit_sequence": response.edit_sequence,
            "watermark_score": response.watermark_score,
            "ratings": response.ratings
        }


class ResponseLoader:
    """Loader for response data saved in TSV format."""

    def __init__(self, filename: str):
        """
        Initialize the ResponseLoader.

        Args:
            filename (str): Path to the TSV file containing response data
        """
        self.filename = Path(filename)
        if not self.filename.exists():
            raise FileNotFoundError(f"File not found: {filename}")

    def _parse_bool(self, value: str) -> bool:
        """Parse string boolean values."""
        return value.lower() == 'true'

    def _parse_list(self, value: str) -> List[str]:
        """Parse string representation of list."""
        try:
            if not value:
                return []
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            logging.warning(f"Failed to parse list value: {value}")
            return []

    def _parse_dict(self, value: str) -> Dict[str, float]:
        """Parse string representation of dictionary."""
        try:
            if not value:
                return {}
            # Try parsing as JSON first
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                # Fall back to ast.literal_eval for other string representations
                return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            logging.warning(f"Failed to parse dictionary value: {value}")
            return {}

    def _parse_float(self, value: str) -> Optional[float]:
        """Parse float value, returning None for empty or invalid values."""
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            logging.warning(f"Failed to parse float value: {value}")
            return None

    def _row_to_response(self, row: Dict[str, str]) -> Response:
        """Convert a row from the TSV file to a Response object."""
        return Response(
            id=int(row['id']),
            text=row['text'],
            prompt=row['prompt'],
            watermark_detected=self._parse_bool(row['watermark_detected']),
            is_watermarked=self._parse_bool(row['is_watermarked']),
            edit_sequence=self._parse_list(row['edit_sequence']),
            watermark_score=self._parse_float(row['watermark_score']),
            ratings=self._parse_dict(row['ratings'])
        )

    def load_all(self) -> List[Response]:
        """
        Load all responses from the file.

        Returns:
            List[Response]: List of all Response objects in the file
        """
        return list(self.iter_responses())

    def iter_responses(self) -> Iterator[Response]:
        """
        Iterate over responses in the file.

        Yields:
            Response: Next Response object from the file
        """
        try:
            with open(self.filename, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file, delimiter='\t')
                for row in reader:
                    try:
                        yield self._row_to_response(row)
                    except Exception as e:
                        logging.error(f"Error parsing row {row.get('id', 'unknown')}: {e}")
                        continue
        except Exception as e:
            logging.error(f"Error reading file {self.filename}: {e}")
            raise

    def load_batch(self, batch_size: int) -> Iterator[List[Response]]:
        """
        Load responses in batches.

        Args:
            batch_size (int): Number of responses to load in each batch

        Yields:
            List[Response]: Next batch of Response objects
        """
        batch = []
        for response in self.iter_responses():
            batch.append(response)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:  # Yield any remaining responses
            yield batch


if __name__ == "__main__":
    # Example usage of the ResponseLoader
    loader = ResponseLoader("evaluation/results/KGW.tsv")
    for batch in loader.load_batch(100):
        print(f"Loaded batch of {len(batch)} responses")
        # Process the batch here
        # For example, you could pass the batch to another pipeline stage
        # and continue processing the responses
        break  # Stop after the first batch for demonstration purposes
    for response in loader.load_all():
        print(response)
        # Process each response here
        # For example, you could analyze the ratings or detection results
        # and generate summary statistics
        break  # Stop after the first response for demonstration purposes
