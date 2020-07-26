# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from .dictionary import Dictionary, TruncatedDictionary

from .fairseq_dataset import FairseqDataset, FairseqIterableDataset

from .base_wrapper_dataset import BaseWrapperDataset

from .append_token_dataset import AppendTokenDataset
from .concat_dataset import ConcatDataset
from .indexed_dataset import IndexedCachedDataset, IndexedDataset, IndexedRawTextDataset, MMapIndexedDataset
from .language_pair_dataset import LanguagePairDataset
from .prepend_token_dataset import PrependTokenDataset
from .strip_token_dataset import StripTokenDataset
from .truncate_dataset import TruncateDataset

from .iterators import (
    CountingIterator,
    EpochBatchIterator,
    GroupedIterator,
    ShardedIterator,
)

__all__ = [
    'AppendTokenDataset',
    'BaseWrapperDataset',
    'ConcatDataset',
    'Dictionary',
    'EpochBatchIterator',
    'FairseqDataset',
    'FairseqIterableDataset',
    'GroupedIterator',
    'IndexedCachedDataset',
    'IndexedDataset',
    'IndexedRawTextDataset',
    'LanguagePairDataset',
    'MMapIndexedDataset',
    'PrependTokenDataset',
    'ShardedIterator',
    'StripTokenDataset',
    'TruncateDataset',
    'TruncatedDictionary',
]