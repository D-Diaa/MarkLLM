# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =======================================================
# text_quality_analyzer.py
# Description: Analyze text quality using various metrics
# =======================================================

import math
import re

import sacrebleu
import torch

from exceptions.exceptions import InvalidAnswerError
from utils.openai_utils import OpenAIAPI


class TextQualityAnalyzer:
    """Base class for text quality analyzer."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str):
        pass

    def analyze_batched(self, texts: list):
        return [self.analyze(text) for text in texts]


class DirectTextQualityAnalyzer(TextQualityAnalyzer):
    """Base class for direct text quality analyzer."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str):
        pass

    def analyze_batched(self, texts: list):
        return [self.analyze(text) for text in texts]


class ReferencedTextQualityAnalyzer(TextQualityAnalyzer):
    """Base class for referenced text quality analyzer."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str, reference):
        pass

    def analyze_batched(self, texts: list, references: list):
        return [self.analyze(text, reference) for text, reference in zip(texts, references)]


class ExternalDiscriminatorTextQualityAnalyzer(TextQualityAnalyzer):
    """Base class for external discriminator text quality analyzer."""

    def __init__(self) -> None:
        pass

    def analyze(self, text1: str, text2: str, description: str):
        pass

    def analyze_batched(self, texts1: list, texts2: list, descriptions: list):
        return [self.analyze(text1, text2, description) for text1, text2, description in zip(texts1, texts2, descriptions)]

class PPLCalculator(DirectTextQualityAnalyzer):
    """Perplexity calculator for text quality analysis."""

    def __init__(self, model, tokenizer, device='cuda') -> None:
        """
            Initialize the perplexity calculator.

            Parameters:
                model: The language model for perplexity calculation.
                tokenizer: The tokenizer for the language model.
                device (str): The device to use for the calculation.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def analyze(self, text: str):
        """Calculate the perplexity of the given text."""
        criterion = torch.nn.CrossEntropyLoss()
        encoded_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(
            self.device)
        logits = self.model(torch.unsqueeze(encoded_text, 0), return_dict=True).logits[0]
        loss = criterion(logits[:-1], encoded_text[1:])
        ppl = torch.exp(loss)
        return ppl.item()


class LogDiversityAnalyzer(DirectTextQualityAnalyzer):
    """Log diversity analyzer for text quality analysis."""

    def __init__(self) -> None:
        super().__init__()

    def _eval_text(self, text: str, ngram: int):
        """Evaluate text to compute the number of unique and total n-grams."""
        tokens = text.split()
        ngram_set = set()
        total_ngrams = 0

        for i in range(len(tokens) - ngram + 1):
            ngram_set.add(" ".join(tokens[i:i + ngram]))
            total_ngrams += 1

        return len(ngram_set), total_ngrams

    def _eval_one_instance(self, text: str, ngram_list: list):
        """Evaluate a single text instance for multiple n-gram lengths."""
        results = {}
        for n in ngram_list:
            unique, total = self._eval_text(text, n)
            results[n] = {"unique": unique, "total": total}
        unique_tokens = set(text.split())
        return results, unique_tokens

    def analyze(self, text: str):
        """Analyze text to compute log diversity based on n-gram uniqueness."""
        ngram_list = [2, 3, 4]
        prediction_results = {n: {"unique": 0, "total": 0} for n in ngram_list}
        unique_token_set = set()

        stripped_text = text.strip()
        ngram_results, unique_tokens = self._eval_one_instance(stripped_text, ngram_list)

        unique_token_set.update(unique_tokens)

        for n in ngram_list:
            prediction_results[n]["unique"] += ngram_results[n]["unique"]
            prediction_results[n]["total"] += ngram_results[n]["total"]

        # Compute diversity scores for each n-gram length
        diversity_scores = [
            1 - (prediction_results[n]["unique"] / prediction_results[n]["total"])
            for n in ngram_list
        ]

        # Overall diversity is the product of individual n-gram diversities
        overall_diversity = (1 - diversity_scores[0] / 100) * (1 - diversity_scores[1] / 100) * (
                    1 - diversity_scores[2] / 100)
        log_diversity = -math.log(max(1 - overall_diversity, math.exp(-20)))

        return log_diversity


class BLEUCalculator(ReferencedTextQualityAnalyzer):
    """BLEU calculator for text quality analysis."""

    def __init__(self) -> None:
        pass

    def analyze(self, text: str, reference: str):
        """Calculate the BLEU score of the given text with the reference."""
        b = sacrebleu.corpus_bleu([text], [[reference]]).score
        return b


class PassOrNotJudger(ReferencedTextQualityAnalyzer):
    """Pass or not judger for text quality analysis."""

    def __init__(self) -> None:
        pass

    def _check_correctness(self, prompt: str, completion: str, test: str, entry_point: str):
        """Check the correctness of the code."""
        check_program = (
                prompt + '\n' + completion + "\n" +
                test + "\n" +
                f"check({entry_point})"
        )
        # print(check_program)
        try:
            exec_globals = {}
            exec(check_program, exec_globals)
            return 1
        except BaseException as e:
            return 0

    def analyze(self, text: str, reference: dict):
        """Check if the text passes the correctness test."""
        passed = self._check_correctness(reference['task'], text, reference['test'], reference['entry_point'])
        return passed


class GPTTextDiscriminator(ExternalDiscriminatorTextQualityAnalyzer):
    """GPT text discriminator for text quality analysis."""

    def __init__(self, openai_model: str, task_description: str) -> None:
        """
            Initialize the GPT text discriminator.

            Parameters:
                openai_model (str): The OpenAI model to use for text discrimination.
                task_description (str): The description of the task for text discrimination.
        """
        self.openai_model = openai_model
        self.task_description = task_description

    def _get_query(self, text1: str, text2: str, question: str):
        """Get the query for text discrimination."""

        query = f"Task Description: {self.task_description}\n"
        query += f"Question: {question}\n"
        query += f"Answer 1: {text1}\n"
        query += f"Answer 2: {text2}\n"
        query += f"Which anwser is better? Only return a number."
        query += f"Return 1 if the first text is better, 2 if the second text is better, 0 if they are equal."
        return query

    def analyze(self, text1: str, text2: str, question: str):
        """Analyze the text to determine which one is better."""
        openai_util = OpenAIAPI(model=self.openai_model, temperature=0.2,
                                system_content="You are a helpful assistant to determine which of the two answers is better based on the given task description.")
        query = self._get_query(text1, text2, question)
        answer = openai_util.get_result(query)
        # validate answer
        if answer not in ['0', '1', '2']:
            raise InvalidAnswerError
        return eval(answer)


class LLMTextRater(ReferencedTextQualityAnalyzer):
    """LLM text discriminator for text quality analysis."""

    def __init__(self, model, tokenizer, device='cuda', cot=True, **kwargs) -> None:
        """
            Initialize the LLM text rater.

            Parameters:
                model: The language model to use for text rating.
                tokenizer: The tokenizer for the language model.
                device (str): The device to use for the rating.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)
        self.system_prompt = """Please act as an impartial judge and evaluate the quality of the response provided by 
        an AI assistant to the user question displayed below. Your evaluation hinges on the following factors: 
        accuracy, level of details, typographical correctness, grammatical correctness and lexical correctness. Be as 
        objective as possible, and account for every one of these criteria in our evaluation. Begin your evaluation 
        by providing a very short explanation. After providing your explanation, please rate the response on a scale of 
        1 to 100 by strictly following this format: "[[rating]]", for example: "Rating: [[51]]". Your rating must come 
        AFTER your explanation, not before.""" if cot else """ You are given a prompt and a response, and you provide a 
        grade out of 100 measuring the quality of the response, in terms of accuracy, level of details, and 
        typographical, grammatical and lexical correctness. Remove points as soon as one of the criteria is missed. 
        please rate the response on a scale of 1 to 100 by strictly following this format: "[[rating]]", 
        for example: "Rating: [[51]]".
        """
        self.instruction = f"Prompt:\n{{}}\nResponse:\n{{}}"
        self.response = "<analysis>" if cot else "Rating: "
        self.num_regex = re.compile(r"([0-9]+\.*[0-9]*)(/100)?")
        self.gen_kwargs = kwargs

    def analyze(self, text: str, reference):
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.instruction.format(reference, text)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        ) + self.response
        # Tokenize the input
        final_input = self.tokenizer([prompt], return_tensors="pt")
        final_input = {k: v.to(self.device) for k, v in final_input.items()}
        # Generate the output
        with torch.inference_mode():
            output = self.model.generate(**final_input, **self.gen_kwargs)
        # Decode the output
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # remove anything before <analysis>
        decoded_output = " ".join(decoded_output.split(self.response)[1:])
        # Extract the rating from the output
        rating = 0.0
        matches = re.findall(self.num_regex, decoded_output)
        if matches and len(matches):
            val = matches[-1][0].replace("[", "").replace("]", "")
            if "/" in val:
                rating = float(val.split("/")[0]) / float(
                    val.split("/")[1]
                )
            else:
                rating = float(val) / 100

        rating = max(min(rating, 1), 0)
        return rating

    def analyze_batched(self, texts: list, references: list):
        prompts =[
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.instruction.format(reference, text)},
                ],
                tokenize=False,
                add_generation_prompt=True,
            ) + self.response
            for text, reference in zip(texts, references)
        ]
        # Tokenize the input
        final_input = self.tokenizer(prompts, return_tensors="pt", padding=True)
        final_input = {k: v.to(self.device) for k, v in final_input.items()}
        # Generate the output
        with torch.inference_mode():
            output = self.model.generate(**final_input, **self.gen_kwargs)
        # Decode the output
        decoded_output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        ratings = []
        for text in decoded_output:
            # remove anything before <analysis>
            text = " ".join(text.split(self.response)[1:])
            # Extract the rating from the output
            rating = 0.0
            matches = re.findall(self.num_regex, text)
            if matches and len(matches):
                val = matches[-1][0].replace("[", "").replace("]", "")
                if "/" in val:
                    rating = float(val.split("/")[0]) / float(
                        val.split("/")[1]
                    )
                else:
                    rating = float(val) / 100

            rating = max(min(rating, 1), 0)
            ratings.append(rating)
        return ratings