# Portable SwissProt experiment code

This archive intentionally contains **code only**. It has no SwissProt CSVs, experiment outputs/checkpoints, or model weights.

All project file paths are relative to this directory. Put the datasets on the new machine at:

```
datasets/Swissprot/
```

The expected CSV names are:

- `cellular_component_trainable.csv`
- `cellular_component_inference_processed.csv`
- `biological_process_trainable.csv`
- `biological_process_inference_processed.csv`
- `molecular_function_trainable.csv`
- `molecular_function_inference_processed.csv`

Create an environment appropriate for your GPU, install the Python packages, then download the model once:

```bash
python -m pip install -r requirements.txt
python -c "from transformers import AutoModel, AutoTokenizer; n='facebook/esm2_t33_650M_UR50D'; AutoTokenizer.from_pretrained(n); AutoModel.from_pretrained(n)"
```

Run the latest baseline configuration with:

```bash
bash run_baseline_normal20000_30000_esm2_650m_lora_all_layers.sh
```

The launcher uses `python` from the active environment and writes outputs under this directory unless `OUTPUT_ROOT` is supplied. The Hugging Face model identifier is intentional; it is not a local filesystem path.

TACS configurations expect optional TrEMBL data under `datasets/TREMBL/`; update the relevant configuration if you store it elsewhere.
