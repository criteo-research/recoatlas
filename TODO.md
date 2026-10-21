# Release TODO

- Validate the optional `grammar` extra and its grammar-dependent tests on a machine with enough
  scratch space. The current dependency lock resolves a full Torch/CUDA stack, so both the uv cache
  and the temporary environment must live on a large filesystem. Verify at minimum:

  ```bash
  UV_CACHE_DIR=/large-scratch/recoatlas-uv-cache \
  UV_PROJECT_ENVIRONMENT=/large-scratch/recoatlas-venv \
    uv sync --extra grammar --frozen

  UV_CACHE_DIR=/large-scratch/recoatlas-uv-cache \
  UV_PROJECT_ENVIRONMENT=/large-scratch/recoatlas-venv \
    uv run --frozen pytest -q tests/test_backend.py
  ```

  This remains unvalidated for the release; the non-grammar tests pass.
