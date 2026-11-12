"""Corpus and tokenizer utilities: the byte-token map (the adapter's bridge into Qwen's embedding table)
and the hnet-kaggle val-split loader (the published 1.4177 protocol's data)."""
import os
import numpy as np


def byte_token_map(tok):
    """byte value -> the id of the tokenizer's SINGLE-BYTE token for it (add_special_tokens off).

    221/256 bytes have one (all printable text bytes); the missing ones are unprintable controls.  The
    map is the bridge that lets a byte-level pipeline reuse the model's own embedding table and LM head:
    a chunk's patch vector is the sum of its bytes' embedding rows, and the head's next-token
    distribution IS a next-byte distribution over the mapped ids.
    """
    m = {}
    for b in range(256):
        ids = tok.encode(bytes([b]).decode("latin-1"), add_special_tokens=False)
        if len(ids) == 1:
            m[b] = ids[0]
    return m


def bytes_to_token_ids(data, bmap, skip_missing=True):
    """Raw bytes -> the list of token ids for the MAPPED bytes.  Returns (ids, coverage) where coverage
    is the fraction of bytes that had a single-byte token.  With skip_missing=False, raises on the
    first unmapped byte."""
    ids, missing = [], 0
    for b in data:
        t = bmap.get(int(b))
        if t is None:
            if not skip_missing:
                raise KeyError(f"byte {b} has no single-byte token")
            missing += 1
            continue
        ids.append(t)
    return ids, 1.0 - missing / max(len(data), 1)


class ValDocs:
    """The hnet-kaggle held-out split (val.bin + val_docs.u32 + val_lens.u32), the published 1.4177 bpb
    protocol's data.  Same format as the hnet repo's tools/eval_ckpt.py Docs class."""

    def __init__(self, d):
        self.data = np.memmap(os.path.join(d, "val.bin"), dtype=np.uint8, mode="r")
        self.offsets = np.fromfile(os.path.join(d, "val_docs.u32"), dtype="<u4")
        self.lengths = np.fromfile(os.path.join(d, "val_lens.u32"), dtype="<u4")

    def __len__(self):
        return len(self.offsets)

    def doc(self, i):
        o, l = int(self.offsets[i]), int(self.lengths[i])
        return self.data[o:o + l]

    def docs(self, n, min_len=2048, seed=0):
        """n document byte-strings, longest-first from a shuffled index (short docs waste chunk slots)."""
        rng = np.random.default_rng(seed)
        idx = [i for i in rng.permutation(len(self)) if self.lengths[i] >= min_len][:n]
        return [bytes(self.doc(i)) for i in idx]
