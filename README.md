# borg-cube
Repository for the borg-cube nlp processing framework

## Inference

All four models use `BorgConfig.eval_batch_size` (default: 32) for prediction.
The tagger, parser, and lemmatizer group sentences by estimated length, tokenize
them in batches, and right-pad only to the longest sequence in each batch.
Returned sentences remain in input order, and empty sentences are preserved.
The tokenizer batches overlapping windows without changing whitespace
reconstruction or overlap scoring.

Each model's `predict()` displays a tqdm progress bar measured in batches.
Pass `show_progress=False` to disable it, for example
`tagger.predict(sentences, show_progress=False)` or
`tokenizer.predict(text, show_progress=False)`. Empty input produces no bar.

Prediction uses PyTorch inference mode and the configured CUDA autocast dtype.
The parser scores dependency relations only for decoded arcs instead of
materializing a relation tensor for every subword pair. Training is unchanged.
Reduce `eval_batch_size` if inference exceeds available device memory; set it
to 1 for single-example inference. It must be positive.

Sentence models still truncate at `max_seq_length`; batching does not add
long-sentence recovery. Annotation preserves token metadata, including
`space_after`, multiword tokens, and empty nodes.
