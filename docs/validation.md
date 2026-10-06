# Validation scope

Validated on 2026-10-06 in a separate Linux CPU environment:

- Python 3.12.13, PyTorch 2.14.1+cpu, Transformers 4.57.1, PEFT 0.17.1,
  Hugging Face Hub 0.36.2, and XGrammar 0.2.6.
- `python -m pytest -q`: **31 passed**, including grammar compilation for
  every requested bundle size from 3 through 7.
- Agent implementation files, strategy, and output-contract source match the
  preserved paper implementation byte-for-byte. Tests also lock the complete
  assembled system prompt for each K, not only the strategy paragraph.
- The judge's system instruction, common guidance, direct-relevance rubric,
  and shuffle seed match the original judge source exactly.
- Actual downloads and checksum verification completed for Musical Instruments,
  Electronics, and H&M. Catalog/reference membership, embedding order, finite
  normalized tensors, and safe checkpoint loading were checked in all three.
- Six real CPU inference smoke runs completed: the deterministic three-tool
  baseline on the first hidden query of each available category, once with
  utility tools and once with semantic tools. Both modes used the pinned real
  retrieval checkpoint/backbone and the corresponding cached embeddings.
- A real saved smoke trace passed judge request validation with **zero API calls**.
- A mocked LLM endpoint exercised the complete frozen-agent → report → SetHit
  → judge workflow, including advisory diversity checking and gold-blind inputs.

These checks validate packaging and the runnable protocol. They are not a new
full benchmark evaluation, a fresh 27B agent run, or a paid hosted-judge run.
The one-query smoke scores are not evidence of aggregate model performance.
No live model endpoint was provisioned or assumed accessible during validation.

Video Games' tool files have pinned source metadata, but its query/reference
file is absent from the pinned publication. It was not included in the six
benchmark smoke runs. No missing reference set or paper result was fabricated.

To repeat the checks, follow the README's installation, artifact verification,
and one-query baseline instructions. To validate a live agent endpoint, run the
one-query LLM command, inspect its report/protocol/API trace, and only then run
the full selected split. For judging, run `--dry-run` first.
