# coding=utf-8

"""Config classes for CPT."""

from __future__ import absolute_import, division, print_function, unicode_literals

from transformers import BartConfig

CPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'fnlp/cpt-base': "https://huggingface.co/fnlp/cpt-base/raw/main/config.json",
    'fnlp/cpt-large': "https://huggingface.co/fnlp/cpt-large/raw/main/config.json",
}

class CPTConfig(BartConfig):
    pretrained_config_archive_map = CPT_PRETRAINED_CONFIG_ARCHIVE_MAP