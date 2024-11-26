import os
from collections import Counter
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from transformers import AutoTokenizer

from evaluation.pipelines.pipeline_stages import ResponseLoader

class TokenFrequencyAnalyzer:
    def __init__(self, tokenizer_name: str = 'bert-base-uncased'):
        """
        Initialize the analyzer with a specific tokenizer.

        Args:
            tokenizer_name (str): Name of the HuggingFace tokenizer to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_texts(self, texts: List[str]) -> List[str]:
        """
        Tokenize a list of texts and return flattened list of tokens.

        Args:
            texts (List[str]): List of text strings to tokenize

        Returns:
            List[str]: Flattened list of tokens
        """
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            all_tokens.extend(tokens)
        return all_tokens

    def get_token_counts(self, texts: List[str]) -> Counter:
        """
        Get token frequency counts for a list of texts.

        Args:
            texts (List[str]): List of text strings to analyze

        Returns:
            Counter: Counter object with token frequencies
        """
        tokens = self.tokenize_texts(texts)
        return Counter(tokens)

    def get_top_tokens_across_collections(
            self,
            text_collections: List[List[str]],
            top_n: int = 50
    ) -> List[str]:
        """
        Get the top N tokens from the union of all text collections.

        Args:
            text_collections (List[List[str]]): List of text collections
            top_n (int): Number of top tokens to return

        Returns:
            List[str]: List of top tokens
        """
        combined_counts = Counter()
        for texts in text_collections:
            counts = self.get_token_counts(texts)
            combined_counts.update(counts)

        top_tokens = [token for token, _ in combined_counts.most_common(top_n)]
        return top_tokens

    def create_comparison_plot(
            self,
            text_collections: List[List[str]],
            collection_labels: List[str] = None,
            top_n: int = 50,
            figsize: Tuple[int, int] = (15, 8),
            colors: List[str] = None,
            plot_type: str = 'bar',  # 'line', 'area', or 'bar'
            normalize: bool = False,
            alpha: float = 0.7,
            style: str = 'default'
    ) -> plt.Figure:
        """
        Create a comparison plot of token frequencies across collections.

        Args:
            text_collections (List[List[str]]): List of text collections to compare
            collection_labels (List[str]): Labels for each collection
            top_n (int): Number of top tokens to display
            figsize (Tuple[int, int]): Figure size
            colors (List[str]): Colors for each collection
            plot_type (str): Type of plot ('line', 'area', or 'bar')
            normalize (bool): Whether to normalize frequencies
            alpha (float): Transparency level
            style (str): Plot style ('default' or 'dark')

        Returns:
            plt.Figure: Matplotlib figure object
        """
        if collection_labels is None:
            collection_labels = [f"Collection {i + 1}" for i in range(len(text_collections))]

        if colors is None:
            colors = sns.color_palette("husl", n_colors=len(text_collections))

        if style == 'dark':
            plt.style.use('dark_background')
        else:
            plt.style.use('default')

        # Get top tokens across all collections
        top_tokens = self.get_top_tokens_across_collections(text_collections[0:1], top_n)

        # Get counts for each collection
        collection_counts = []
        for texts in text_collections:
            counts = self.get_token_counts(texts)
            token_counts = [counts[token] for token in top_tokens]

            if normalize:
                total = sum(token_counts)
                token_counts = [count / total for count in token_counts] if total > 0 else token_counts

            collection_counts.append(token_counts)

        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(top_tokens))
        width = 0.8 / len(text_collections)  # Width of bars

        if plot_type == 'bar':
            # Create grouped bar chart
            for i, (counts, label, color) in enumerate(zip(collection_counts, collection_labels, colors)):
                ax.bar(x + i * width - (len(text_collections) - 1) * width / 2,
                       counts,
                       width,
                       label=label,
                       color=color,
                       alpha=alpha)
        elif plot_type == 'line':
            for counts, label, color in zip(collection_counts, collection_labels, colors):
                ax.plot(x, counts, label=label, color=color, alpha=alpha,
                        linewidth=2, marker='o', markersize=4)
        else:  # area
            for counts, label, color in zip(collection_counts, collection_labels, colors):
                ax.fill_between(x, counts, alpha=alpha, label=label, color=color)

        # Customize the plot
        ax.set_xticks(x)
        #x-axis font size
        plt.xticks(fontsize=16)
        #y-axis font size
        plt.yticks(fontsize=16)


        # Clean up token display
        display_tokens = []
        for token in top_tokens:
            # Replace Ġ with '▁' (more visible space marker) or other special chars
            cleaned_token = token.replace('Ġ', '▁')
            # Add quotes around tokens for clarity
            display_tokens.append(f"'{cleaned_token}'")

        # Adjust label positioning and rotation
        ax.set_xticklabels(display_tokens, rotation=90, ha='center', va='top')

        # Add some padding at the bottom for the labels
        plt.subplots_adjust(bottom=0.2)

        ax.set_xlabel('Tokens', fontsize=16)
        ax.set_ylabel('Frequency' if not normalize else 'Normalized Frequency', fontsize=16)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=16)

        # Add subtle grid lines
        ax.grid(True, linestyle='--', alpha=0.2)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        return fig


#%%
directory = "evaluation/results"
top_n = 50
file = "Unigram_new_reevaluated123456.tsv"
filename = os.path.join(directory, file)
loader = ResponseLoader(filename)
responses = loader.load_all()
labels = set([response.edit_sequence[0] for response in responses if response.is_watermarked])
#%%
base = ['none']
against = [
    'Paraphrase(models/dpo_qwen_3_exponential)_0',
    'Paraphrase(models/Unigram_new/Qwen/Qwen2.5-3B-Instruct)_1',
    'Paraphrase(models/dpo_qwen_3_exponential)_1',
    'Paraphrase(GPT4o)',
    'Paraphrase(GPT3.5)'
]
names = ['Qwen-3b', 'Ours-Qwen-3b-Unigram', 'Ours-Qwen-3b-Exp', 'GPT4o', 'GPT3.5']
base_collection = [response.text.strip() for response in responses if response.edit_sequence == base]
against_collections = [
    [response.text.strip() for response in responses if response.edit_sequence == [label]]
    for label in against
]
against_labels = names
base_name = 'Original'
#%%
# Create analyzer
analyzer = TokenFrequencyAnalyzer('meta-llama/Llama-3.1-8B-Instruct')
colors = sns.color_palette("tab10", n_colors=len(labels))
for i, collection in enumerate(against_collections):
    labels = [base_name, against_labels[i]]
    collections = [base_collection, collection]
    # Create bar plot
    fig = analyzer.create_comparison_plot(
        collections,
        collection_labels=labels,
        colors=colors,
        top_n=top_n,
        plot_type='bar',
        figsize=(20, 10),
        alpha=1
    )
    plt.figure(fig.number)
    plt.tight_layout()
    name = f"{file.replace('.tsv', '')}_{against_labels[i]}_token_frequency.pdf"
    plt.savefig(os.path.join(directory, name))
    plt.show()
