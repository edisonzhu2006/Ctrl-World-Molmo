"""
Batch sampler that guarantees each minibatch contains samples from two domains
(e.g. real robot + simulation), by yielding encoded indices consumed by
`Dataset_mix` when `args.dataset_mix_batch` is True.

Encoded index layout: ``dataset_id * stride + local_row`` where
``stride > max(len(samples))`` for all sources (see ``Dataset_mix._mix_index_stride``).
"""

from __future__ import annotations

import random
import re
from typing import Any, List


def parse_dataset_mix_counts(value: str, batch_size: int) -> List[int]:
    if not str(value).strip():
        raise ValueError("dataset_mix_counts is empty")
    parts = [int(x.strip()) for x in re.split(r"[,+]", str(value)) if str(x).strip()]
    if len(parts) != 2:
        raise ValueError(
            f"dataset_mix_counts must have exactly two integers (counts per dataset in "
            f"dataset_names order), got {parts!r} from {value!r}"
        )
    if parts[0] + parts[1] != batch_size:
        raise ValueError(
            f"dataset_mix_counts must sum to train_batch_size={batch_size}, got {parts[0]}+{parts[1]}"
        )
    if parts[0] <= 0 or parts[1] <= 0:
        raise ValueError(f"Each count must be positive, got {parts}")
    return parts


class MixedDomainBatchSampler:
    """
    Yields batches of encoded indices. Each batch has ``counts[0]`` rows from
    dataset_id 0 and ``counts[1]`` rows from dataset_id 1; order within the batch
    is shuffled. Indices are drawn uniformly at random with replacement per slot.
    """

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        counts: List[int],
        *,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        if not getattr(dataset, "dataset_mix_batch", False):
            raise ValueError("dataset.dataset_mix_batch must be True to use MixedDomainBatchSampler")
        if len(dataset.samples_len) != 2:
            raise ValueError("MixedDomainBatchSampler currently supports exactly two datasets")
        if len(counts) != 2 or counts[0] + counts[1] != batch_size:
            raise ValueError("counts must be length 2 and sum to batch_size")

        self.dataset = dataset
        self.batch_size = batch_size
        self.counts = list(counts)
        self.seed = int(seed)
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank) % self.num_replicas

        self.stride = int(dataset._mix_index_stride)
        self.lens = [int(x) for x in dataset.samples_len]
        self.max_id = int(dataset.max_id)
        self.num_batches = (self.max_id + self.batch_size - 1) // self.batch_size
        self._epoch_idx = 0
        self._resume_batch_idx = 0
        self._consume_resume_offset = False

    def __len__(self) -> int:
        return (self.num_batches + self.num_replicas - 1) // self.num_replicas

    def state_dict(self) -> dict:
        return {
            "kind": "mixed_domain_batch_sampler",
            "epoch_idx": self._epoch_idx,
            "batch_idx": self._resume_batch_idx,
            "seed": self.seed,
        }

    def load_state_dict(self, state: dict) -> None:
        self._epoch_idx = int(state["epoch_idx"])
        self._resume_batch_idx = int(state["batch_idx"])
        self.seed = int(state["seed"])
        self._consume_resume_offset = True

    def __iter__(self):
        epoch = self._epoch_idx
        self._epoch_idx += 1
        skip = self._resume_batch_idx if self._consume_resume_offset else 0
        if self._consume_resume_offset:
            self._consume_resume_offset = False
        yielded = 0
        for batch_idx in range(self.rank, self.num_batches, self.num_replicas):
            if yielded < skip:
                yielded += 1
                continue
            rng = random.Random(self.seed + epoch * 1_000_003 + batch_idx)
            batch: List[int] = []
            for _ in range(self.counts[0]):
                local = rng.randrange(self.lens[0])
                batch.append(0 * self.stride + local)
            for _ in range(self.counts[1]):
                local = rng.randrange(self.lens[1])
                batch.append(1 * self.stride + local)
            rng.shuffle(batch)
            yielded += 1
            yield batch
