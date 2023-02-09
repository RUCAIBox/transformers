# coding=utf-8
"""Tokenization classes for UniLM."""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from transformers import BertTokenizer

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {'vocab_file': 'vocab.txt'}

PRETRAINED_VOCAB_FILES_MAP = {
    'vocab_file':
    {
        'microsoft/unilm-large-cased': "https://huggingface.co/microsoft/unilpm-large-cased/raw/main/vocab.txt",
        'microsoft/unilm-base-cased': "https://huggingface.co/microsoft/unilm-base-cased/raw/main/vocab.txt"
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'microsoft/unilm-large-cased': 512,
    'microsoft/unilm-base-cased': 512
}


class UnilmTokenizer(BertTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
