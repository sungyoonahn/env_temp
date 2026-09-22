import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import warnings
import train as training_module
from transformers import BertModel, BertTokenizer
from transformers import AutoModel, AutoTokenizer


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', '-b', default=32, help='batch size', type=int)
parser.add_argument('--input_csv', default=["datasets/inference/new_inference_csv"], nargs='+', help='One or more paths to input CSV files or directories used for CC, BP, and MF inference', type=str)
parser.add_argument('--output_dir', default="results/", help='Directory to save visualizations and result files', type=str)
parser.add_argument('--model', default="all", help='Model to use: ProtBERT, ESM2_650M, ESM2_3B, or all', type=str)

args = parser.parse_args()

# Model configurations
MODEL_CONFIGS = {
    "ProtBERT": {
        "model_name": "Rostlab/prot_bert_bfd",
        "use_auto_model": False,
        "cc_weights": "ProtBERT_results/cellular_component_1013_13600/best_weights.pth",
        "bp_weights": "ProtBERT_results/biological_process_1013_44800/best_weights.pth",
        "mf_weights": "ProtBERT_results/molecular_function_1013_6800/best_weights.pth",
    },
    "ESM2_650M": {
        "model_name": "facebook/esm2_t33_650M_UR50D",
        "use_auto_model": True,
        "cc_weights": "ESM2_650M_results/cellular_component_0112_13600/best_weights.pth",
        "bp_weights": "ESM2_650M_results/biological_process_0107_48000/best_weights.pth",
        "mf_weights": "ESM2_650M_results/molecular_function_0106_6800/best_weights.pth",
    },
    "ESM2_3B": {
        "model_name": "facebook/esm2_t36_3B_UR50D",
        "use_auto_model": True,
        "cc_weights": "ESM2_3B_results/cellular_component_0117_13600/best_weights.pth",
        "bp_weights": "ESM2_3B_results/biological_process_0113_48000/best_weights.pth",
        "mf_weights": "ESM2_3B_results/molecular_function_0119_6800/best_weights.pth",
    }
}


def tsne_plot(save_dir_path, features, pred_labels, bacteria_name, type, n_classes, class_names, model_name="model"):
    torch.cuda.empty_cache()

    n_samples = len(features)
    perplexity = min(10, max(1, (n_samples - 1) // 3))
    tsne_result = TSNE(n_components=2, perplexity=perplexity).fit_transform(features)
    unique_labels = np.unique(pred_labels)

    cmap_name = "Set3"
    try:
        cmap = plt.cm.get_cmap(cmap_name, n_classes)
    except ValueError:
        print(f"[WARN] Colormap '{cmap_name}' not found. Skipping.")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=pred_labels, cmap=cmap)

    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=cmap(i / (n_classes - 1 if n_classes > 1 else 1)),
                          markersize=10) for i in unique_labels]
    legend_labels = [class_names[i] for i in unique_labels if i < len(class_names)]

    plt.legend(handles, legend_labels, title="Classes", loc='lower right')
    plt.title(f"t-SNE Visualization of {type} Classification Results for {bacteria_name}")
    plt.tight_layout()

    save_file_path = os.path.join(save_dir_path, f"{bacteria_name}_{type}_{model_name}_Set3.png")
    plt.savefig(save_file_path)
    plt.close()


def load_infer_df(csv_path, needs_spaces):
    infer_df = pd.read_csv(csv_path)
    if 'tensor id' not in infer_df.columns:
        infer_df.insert(0, 'tensor id', range(len(infer_df)))
    infer_df["Sequence"] = infer_df["Sequence"].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)
    if needs_spaces:
        infer_df['Sequence'] = infer_df.apply(lambda row: " ".join(row["Sequence"]), axis=1)
    return infer_df


def make_loader(infer_df, tokenizer):
    dummy_labels = np.zeros(len(infer_df), dtype=np.int64)
    ds = training_module.CustomProteinDataset(infer_df["Sequence"], dummy_labels, tokenizer, infer_df["tensor id"])
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=training_module.collate_fn)


def build_merged_csv(infer_df, infer_results, class_names, pred_col_name):
    """Merge original infer_df with inference results; move pred value column to front."""
    n = len(class_names)
    infer_results = infer_results.copy()
    infer_results.columns = [str(c) for c in infer_results.columns]
    infer_results = infer_results.rename(columns={str(i): class_names[i] for i in range(n)})
    prob_cols = [c for c in infer_results.columns if c != 'tensor id']
    infer_results[pred_col_name] = infer_results[prob_cols].idxmax(axis=1)

    merged = pd.merge(infer_df.copy(), infer_results, on='tensor id', how='left')
    # Drop label column if present (no ground truth in new inference)
    if 'label' in merged.columns:
        merged.drop(columns=['label'], inplace=True)
    # Move pred value to front
    other_cols = [c for c in merged.columns if c != pred_col_name]
    merged = merged[[pred_col_name] + other_cols]
    return merged


if __name__ == "__main__":
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    if args.model == "all":
        models_to_run = list(MODEL_CONFIGS.keys())
    elif args.model in MODEL_CONFIGS:
        models_to_run = [args.model]
    else:
        raise ValueError(f"Model must be one of {list(MODEL_CONFIGS.keys())} or 'all'. Got: {args.model}")

    device = torch.device('cuda')
    training_module.args = args
    training_module.device = device

    os.makedirs(args.output_dir, exist_ok=True)

    def collect_csvs(paths):
        result = []
        for path in paths:
            if os.path.isdir(path):
                result.extend(sorted(os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.csv')))
            elif os.path.isfile(path):
                result.append(path)
            else:
                raise FileNotFoundError(f"Input path not found: {path}")
        return result

    cc_class_names = [
        "CC normal",
        "CC Flagellum|Bacterial flagellum",
        "CC Cell outer membrane",
        "CC Cell wall",
        "CC Fimbrium",
        "CC Secreted"
    ]
    bp_class_names = [
        "BP normal",
        "BP Antibiotic resistance|Trimethoprim resistance",
        "BP Metal Resistance",
        "BP Toxin-antitoxin system",
        "BP Virulence"
    ]
    mf_class_names = [
        "MF normal",
        "MF Toxin",
        "MF Bacteriotoxin"
    ]

    csv_paths = collect_csvs(args.input_csv)

    for model_name in models_to_run:
        model_config = MODEL_CONFIGS[model_name]
        print(f"\n{'='*60}")
        print(f"Running inference with model: {model_name}")
        print(f"{'='*60}")

        if model_config['use_auto_model']:
            base_model = AutoModel.from_pretrained(model_config['model_name'], output_hidden_states=True, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'], do_lower_case=False, trust_remote_code=True)
        else:
            base_model = BertModel.from_pretrained(model_config['model_name'], output_hidden_states=True)
            tokenizer = BertTokenizer.from_pretrained(model_config['model_name'], do_lower_case=False)

        training_module.tokenizer = tokenizer
        needs_spaces = not model_config['use_auto_model']
        cc_bw_path = model_config['cc_weights']
        bp_bw_path = model_config['bp_weights']
        mf_bw_path = model_config['mf_weights']

        lora_cfg = LoraConfig(inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["query", "value", "key"])

        for csv_path in csv_paths:
            item = os.path.basename(csv_path)
            print(f"\nProcessing input CSV: {item}")
            base_name = os.path.splitext(item)[0]

            bacteria_dir = os.path.join(args.output_dir, base_name)
            os.makedirs(bacteria_dir, exist_ok=True)

            # ----- CC -----
            print("Loading Cellular Component Best Weights ... ")
            infer_df = load_infer_df(csv_path, needs_spaces)
            CC_model = training_module.CustomProtBERTModel(get_peft_model(base_model, lora_cfg), 6)
            CC_model.to(device)
            CC_model = training_module.load_model(CC_model, cc_bw_path)
            CC_infer_results, CC_features, _ = training_module.infer_4_chosun(CC_model, make_loader(infer_df, tokenizer), bacteria_dir)
            cc_pred_labels = CC_infer_results.drop(columns=['tensor id']).idxmax(axis=1).to_numpy()
            tsne_plot(bacteria_dir, CC_features, cc_pred_labels, base_name, "CC", 6, cc_class_names, model_name=model_name)
            cc_merged = build_merged_csv(infer_df, CC_infer_results, cc_class_names, 'CC pred value')
            cc_merged.to_csv(os.path.join(bacteria_dir, f"{base_name}_{model_name}_CC_results.csv"), index=False)
            print(f"\nCC Predictions:"); print(pd.Series(cc_pred_labels).value_counts())

            # ----- BP -----
            print("Loading Biological Process Best Weights ... ")
            infer_df = load_infer_df(csv_path, needs_spaces)
            BP_model = training_module.CustomProtBERTModel(get_peft_model(base_model, lora_cfg), 5)
            BP_model.to(device)
            BP_model = training_module.load_model(BP_model, bp_bw_path)
            BP_infer_results, BP_features, _ = training_module.infer_4_chosun(BP_model, make_loader(infer_df, tokenizer), bacteria_dir)
            bp_pred_labels = BP_infer_results.drop(columns=['tensor id']).idxmax(axis=1).to_numpy()
            tsne_plot(bacteria_dir, BP_features, bp_pred_labels, base_name, "BP", 5, bp_class_names, model_name=model_name)
            bp_merged = build_merged_csv(infer_df, BP_infer_results, bp_class_names, 'BP pred value')
            bp_merged.to_csv(os.path.join(bacteria_dir, f"{base_name}_{model_name}_BP_results.csv"), index=False)
            print(f"\nBP Predictions:"); print(pd.Series(bp_pred_labels).value_counts())

            # ----- MF -----
            print("Loading Molecular Function Best Weights ... ")
            infer_df = load_infer_df(csv_path, needs_spaces)
            MF_model = training_module.CustomProtBERTModel(get_peft_model(base_model, lora_cfg), 3)
            MF_model.to(device)
            MF_model = training_module.load_model(MF_model, mf_bw_path)
            MF_infer_results, MF_features, _ = training_module.infer_4_chosun(MF_model, make_loader(infer_df, tokenizer), bacteria_dir)
            mf_pred_labels = MF_infer_results.drop(columns=['tensor id']).idxmax(axis=1).to_numpy()
            tsne_plot(bacteria_dir, MF_features, mf_pred_labels, base_name, "MF", 3, mf_class_names, model_name=model_name)
            mf_merged = build_merged_csv(infer_df, MF_infer_results, mf_class_names, 'MF pred value')
            mf_merged.to_csv(os.path.join(bacteria_dir, f"{base_name}_{model_name}_MF_results.csv"), index=False)
            print(f"\nMF Predictions:"); print(pd.Series(mf_pred_labels).value_counts())

    print("\nAll inference complete!")
