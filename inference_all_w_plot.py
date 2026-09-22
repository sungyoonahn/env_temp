import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.nn import CrossEntropy , MSELoss
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, auc, precision_recall_curve

from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassMatthewsCorrCoef, MulticlassAUROC
import warnings
import datetime
import train as training_module
import data_utils
from transformers import BertModel, BertTokenizer
from transformers import AutoModel, AutoTokenizer


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--datasets_dir', '-d', default="datasets/", help='input dir', type=str)
parser.add_argument('--results_dir', '-r', default="results/", help='output dir', type=str)
parser.add_argument('--n_classes', '-n', default=6, help='number of classes', type=int)
parser.add_argument("--lora", default=True, help='true or false for lora', type=bool)
parser.add_argument('--batch_size', '-b', default=1, help='batch size', type=int)
parser.add_argument('--learning_rate', '-l', default=2e-5, help='learning rate', type=float)
parser.add_argument('--epoch', '-e', default=20, help='epoch', type=int)
parser.add_argument('--local_save_path', '-sv', default='/', help="local save path", type=str)
parser.add_argument('--fine_tune', '-ft', default=False, help='true or false for best weights', type=bool)
parser.add_argument('--mode', '-m', default="train", help="Choose mode, either train or instance")
parser.add_argument('--cc-input-csv', default='datasets/Swissprot/cellular_component_inference_processed.csv', help='Swiss-Prot CC inference CSV.', type=str)
parser.add_argument('--bp-input-csv', default='datasets/Swissprot/biological_process_inference_processed.csv', help='Swiss-Prot BP inference CSV.', type=str)
parser.add_argument('--mf-input-csv', default='datasets/Swissprot/molecular_function_inference_processed.csv', help='Swiss-Prot MF inference CSV.', type=str)
parser.add_argument('--results_dir_base', default='results/swissprot_encoder_tsne_inference', help='Directory for Swiss-Prot inference tables and encoder t-SNE maps.', type=str)
parser.add_argument('--models', nargs='+', default=["ProtBERT", "ESM2_650M", "ESM2_3B"], help='Models to run: ProtBERT, ESM2_650M, or ESM2_3B', type=str)
parser.add_argument('--tsne-perplexities', default='100,150,200', help='Comma-separated t-SNE perplexities; empty disables plotting.')
args = parser.parse_args()
args.tsne_perplexities = [int(value) for value in args.tsne_perplexities.split(',') if value.strip()]

# Grab every available matplotlib colormap once so all styles are rendered.
DEFAULT_T_SNE_COLORMAPS = sorted(plt.colormaps())

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

def tsne_plot(save_dir_path, features, true_labels, bacteria_name, task, class_names, model_name, perplexities):
    """Save t-SNE maps from the loaded model's final encoder embeddings."""
    if not perplexities:
        return
    features = np.asarray(features, dtype=np.float32)
    true_labels = np.asarray(true_labels, dtype=np.int64)
    if len(features) != len(true_labels):
        raise ValueError(f"Feature/label mismatch for {model_name}/{task}.")
    valid_perplexities = [value for value in perplexities if 0 < value < len(features)]
    if not valid_perplexities:
        raise ValueError(f"No valid t-SNE perplexities for {len(features)} samples.")
    components = min(50, features.shape[0] - 1, features.shape[1])
    reduced_features = PCA(n_components=components, random_state=42).fit_transform(features)
    colors = ["#A0A7B4", "#E45756", "#4C78A8", "#F2CF5B", "#54A24B", "#B279A2"]
    rows = []
    for perplexity in valid_perplexities:
        coordinates = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", init="pca", random_state=42).fit_transform(reduced_features)
        rows.append({"perplexity": perplexity, "trustworthiness_at_12": trustworthiness(reduced_features, coordinates, n_neighbors=min(12, len(features) - 1))})
        figure, axis = plt.subplots(figsize=(10, 7.5), dpi=180)
        for label, class_name in enumerate(class_names):
            selected = true_labels == label
            if selected.any():
                axis.scatter(coordinates[selected, 0], coordinates[selected, 1], s=8 if label == 0 else 26, alpha=0.28 if label == 0 else 0.88, color=colors[label], edgecolors="none", label=f"{class_name} (n={selected.sum():,})")
        axis.set(title=f"{task} inference: {model_name} encoder t-SNE (perplexity {perplexity})", xlabel="t-SNE 1", ylabel="t-SNE 2")
        axis.legend(title="Original inference label", fontsize=8, loc="best")
        figure.tight_layout()
        figure.savefig(os.path.join(save_dir_path, f"{bacteria_name}_{task}_{model_name}_encoder_tsne_perplexity_{perplexity:03d}.png"), bbox_inches="tight")
        plt.close(figure)
    pd.DataFrame(rows).sort_values(["trustworthiness_at_12", "perplexity"], ascending=[False, True]).to_csv(os.path.join(save_dir_path, f"{bacteria_name}_{task}_{model_name}_encoder_tsne_perplexity_comparison.csv"), index=False)

def compute_metrics(true_labels, pred_labels, probs, n_classes):
    """Compute classification metrics: Accuracy, F1, MCC, AUROC, AUPRC and confusion matrix"""
    # Ensure pred_labels are integers
    if isinstance(pred_labels, np.ndarray):
        if pred_labels.dtype == object:
            pred_labels = np.array([int(p) if isinstance(p, (int, np.integer)) else p for p in pred_labels])
        pred_labels = pred_labels.astype(np.int64)
    
    # Convert to tensors
    true_tensor = torch.tensor(true_labels, dtype=torch.long, device=device)
    pred_tensor = torch.tensor(pred_labels, dtype=torch.long, device=device)
    
    # Initialize metrics on device
    acc_metric = MulticlassAccuracy(num_classes=n_classes, average='macro').to(device)
    f1_metric = MulticlassF1Score(num_classes=n_classes, average='macro').to(device)
    mcc_metric = MulticlassMatthewsCorrCoef(num_classes=n_classes).to(device)
    auroc_metric = MulticlassAUROC(num_classes=n_classes, average='macro').to(device)
    
    # Compute metrics
    acc = acc_metric(pred_tensor, true_tensor).item()
    f1 = f1_metric(pred_tensor, true_tensor).item()
    mcc = mcc_metric(pred_tensor, true_tensor).item()
    
    # AUROC with probabilities
    if probs is not None:
        if isinstance(probs, np.ndarray) and probs.dtype == object:
            probs = probs.astype(np.float32)
        probs_tensor = torch.tensor(probs, dtype=torch.float32, device=device)
        auroc = auroc_metric(probs_tensor, true_tensor).item()
    else:
        auroc = None
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(n_classes)))
    
    # Calculate AUPRC (Area Under Precision-Recall Curve) for each class and average
    auprc_scores = []
    for i in range(n_classes):
        true_binary = (true_labels == i).astype(int)
        if len(np.unique(true_binary)) > 1:
            precision, recall, _ = precision_recall_curve(true_binary, probs[:, i] if probs is not None else (pred_labels == i).astype(float))
            auprc = auc(recall, precision)
            auprc_scores.append(auprc)
    auprc = np.mean(auprc_scores) if auprc_scores else None
    
    return {
        'Accuracy': acc,
        'F1 Score': f1,
        'MCC': mcc,
        'AUROC': auroc,
        'AUPRC': auprc,
        'Confusion Matrix': cm
    }

def merge_results_by_bacteria(results_dir_base):
    """Merge all results for each bacteria into a single Excel file"""
    bacteria_folders = [d for d in os.listdir(results_dir_base)
                       if os.path.isdir(os.path.join(results_dir_base, d))]

    for bacteria_name in bacteria_folders:
        bacteria_path = os.path.join(results_dir_base, bacteria_name)
        csv_files = [f for f in os.listdir(bacteria_path) if f.endswith('_results.csv')]

        if not csv_files:
            print(f"No results CSV files found for {bacteria_name}")
            continue

        # Read all CSVs
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(os.path.join(bacteria_path, csv_file))
            dfs.append(df)

        # Merge on tensor id and Sequence
        if dfs:
            merged_df = dfs[0]
            for df in dfs[1:]:
                merged_df = pd.merge(merged_df, df, on=['tensor id', 'Sequence'], how='outer')

            # Save merged results
            excel_path = os.path.join(bacteria_path, f"{bacteria_name}_merged_results.xlsx")
            merged_df.to_excel(excel_path, sheet_name='Merged Results', index=False)
            print(f"Saved merged results: {excel_path}")

if __name__ == "__main__":
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Validate models
    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model must be one of {list(MODEL_CONFIGS.keys())}. Got: {model_name}")

    print(f"\nRunning inference for models: {args.models}")

    # initialize device - cuda if available cpu if else
    device = torch.device('cuda')

    # Set globals for the training_module
    training_module.args = args
    training_module.device = device

    # 2 . Inference Part
    print("Running inference ...")
    print("Processing all bacteria CSV files with all specified models ...")

    print("Inference Mode")
    # Ensure output directory exists
    os.makedirs(args.results_dir_base, exist_ok=True)

    def collect_csvs(path):
        """Collect all CSV files from a directory"""
        if os.path.isdir(path):
            return sorted([os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.csv')])
        raise FileNotFoundError(f"CSV directory not found: {path}")
    
    # Class names for CC, BP, MF
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

    # Get all bacteria CSV files
    csv_files = [None]
    print('Using task-specific Swiss-Prot inference CSVs for CC, BP, and MF.')

    # Loop through each model
    for model_name in args.models:
        print(f"\n{'='*80}")
        print(f"Processing model: {model_name}")
        print(f"{'='*80}")

        model_config = MODEL_CONFIGS[model_name]
        print(f"Model name: {model_config['model_name']}")

        # Initialize model and tokenizer for this model
        if model_config['use_auto_model']:
            # Use AutoModel for ESM2 models
            model = AutoModel.from_pretrained(model_config['model_name'], output_hidden_states=True, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'], do_lower_case=False, trust_remote_code=True)
        else:
            # Use BertModel for ProtBERT
            model = BertModel.from_pretrained(model_config['model_name'], output_hidden_states=True)
            tokenizer = BertTokenizer.from_pretrained(model_config['model_name'], do_lower_case=False)

        training_module.tokenizer = tokenizer

        # Get weight file paths from config
        cc_bw_path = model_config['cc_weights']
        bp_bw_path = model_config['bp_weights']
        mf_bw_path = model_config['mf_weights']

        # Determine if spaces are needed (only for ProtBERT)
        needs_spaces = not model_config['use_auto_model']

        # Loop through each bacteria CSV
        for _ in csv_files:
            bacteria_name = 'SwissProt_inference'
            print(f"\n--- Processing {bacteria_name} ---")

            # Create results directory for this bacteria
            bacteria_results_dir = os.path.join(args.results_dir_base, bacteria_name)
            os.makedirs(bacteria_results_dir, exist_ok=True)

            # ----- CC -----
            print("  Running CC inference...")
            infer_df = pd.read_csv(args.cc_input_csv)
            print(infer_df['label'].value_counts())
            infer_df["Sequence"] = infer_df["Sequence"].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)
            if needs_spaces:
                infer_df['Sequence'] = infer_df.apply(lambda row: " ".join(row["Sequence"]), axis=1)
            infer_dataset = training_module.CustomProteinDataset(infer_df["Sequence"], infer_df["label"], tokenizer, infer_df["tensor id"])
            infer_loader = DataLoader(infer_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=training_module.collate_fn)

            config = LoraConfig(
                inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["query", "value", "key"]
            )
            CC_model = get_peft_model(model, config)
            training_module.print_trainable_parameters(CC_model)
            CC_model = training_module.CustomProtBERTModel(CC_model, 6)
            training_module.print_trainable_parameters(CC_model)
            CC_model.to(device)

            print(f"    Loading CC weights from {cc_bw_path}")
            CC_model = training_module.load_model(CC_model, cc_bw_path)
            CC_model.to(device)

            CC_infer_results, CC_features, _ = training_module.infer_4_chosun(CC_model, infer_loader, bacteria_results_dir)
            cc_pred_labels = CC_infer_results.drop(columns=['tensor id']).idxmax(axis=1).to_numpy()

            cc_probs = CC_infer_results[[col for col in CC_infer_results.columns if col != 'tensor id']].values.astype(np.float32)
            CC_infer_results.columns = [str(col) for col in CC_infer_results.columns]
            tsne_plot(bacteria_results_dir, CC_features, infer_df['label'].to_numpy(), bacteria_name, "CC", cc_class_names, model_name, args.tsne_perplexities)
            prob_cols = [col for col in CC_infer_results.columns if col != 'tensor id']
            CC_infer_results['CC pred value'] = CC_infer_results[prob_cols].idxmax(axis=1)

            merged_df = pd.merge(infer_df.copy(), CC_infer_results, on='tensor id', how='left')
            true_labels_array = infer_df['label'].values
            merged_df['True Label'] = merged_df['label'].map({i: cc_class_names[i] for i in range(len(cc_class_names))})
            merged_df['CC Misclassified'] = ['Yes' if true_labels_array[i] != cc_pred_labels[i] else 'No' for i in range(len(true_labels_array))]

            cols_to_move = ['CC pred value']
            other_cols = [col for col in merged_df.columns if col not in cols_to_move]
            merged_df = merged_df[cols_to_move + other_cols]
            merged_df.drop(columns=['label'], inplace=True)

            print(f"    CC Predictions: {pd.Series(cc_pred_labels).value_counts().to_dict()}")
            cc_metrics = compute_metrics(infer_df['label'].values, cc_pred_labels, cc_probs, 6)

            # Save CC results CSV (keep tensor id and Sequence for merging)
            cc_results_csv = os.path.join(bacteria_results_dir, f"{bacteria_name}_{model_name}_CC_results.csv")
            merged_df.to_csv(cc_results_csv, index=False)

            # ----- BP -----
            print("  Running BP inference...")
            infer_df = pd.read_csv(args.bp_input_csv)
            print(infer_df['label'].value_counts())
            infer_df["Sequence"] = infer_df["Sequence"].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)
            if needs_spaces:
                infer_df['Sequence'] = infer_df.apply(lambda row: " ".join(row["Sequence"]), axis=1)
            infer_dataset = training_module.CustomProteinDataset(infer_df["Sequence"], infer_df["label"], tokenizer, infer_df["tensor id"])
            infer_loader = DataLoader(infer_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=training_module.collate_fn)

            config = LoraConfig(
                inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["query", "value", "key"]
            )
            BP_model = get_peft_model(model, config)
            training_module.print_trainable_parameters(BP_model)
            BP_model = training_module.CustomProtBERTModel(BP_model, 5)
            training_module.print_trainable_parameters(BP_model)
            BP_model.to(device)

            print(f"    Loading BP weights from {bp_bw_path}")
            BP_model = training_module.load_model(BP_model, bp_bw_path)
            BP_model.to(device)

            BP_infer_results, BP_features, _ = training_module.infer_4_chosun(BP_model, infer_loader, bacteria_results_dir)
            bp_pred_labels = BP_infer_results.drop(columns=['tensor id']).idxmax(axis=1).to_numpy()

            bp_probs = BP_infer_results[[col for col in BP_infer_results.columns if col != 'tensor id']].values.astype(np.float32)
            BP_infer_results.columns = [str(col) for col in BP_infer_results.columns]
            tsne_plot(bacteria_results_dir, BP_features, infer_df['label'].to_numpy(), bacteria_name, "BP", bp_class_names, model_name, args.tsne_perplexities)
            prob_cols = [col for col in BP_infer_results.columns if col != 'tensor id']
            BP_infer_results['BP pred value'] = BP_infer_results[prob_cols].idxmax(axis=1)

            merged_df = pd.merge(infer_df.copy(), BP_infer_results, on='tensor id', how='left')
            true_labels_array = infer_df['label'].values
            merged_df['True Label'] = merged_df['label'].map({i: bp_class_names[i] for i in range(len(bp_class_names))})
            merged_df['BP Misclassified'] = ['Yes' if true_labels_array[i] != bp_pred_labels[i] else 'No' for i in range(len(true_labels_array))]

            cols_to_move = ['BP pred value']
            other_cols = [col for col in merged_df.columns if col not in cols_to_move]
            merged_df = merged_df[cols_to_move + other_cols]
            merged_df.drop(columns=['label'], inplace=True)

            print(f"    BP Predictions: {pd.Series(bp_pred_labels).value_counts().to_dict()}")
            bp_metrics = compute_metrics(infer_df['label'].values, bp_pred_labels, bp_probs, 5)

            # Save BP results CSV (keep tensor id and Sequence for merging)
            bp_results_csv = os.path.join(bacteria_results_dir, f"{bacteria_name}_{model_name}_BP_results.csv")
            merged_df.to_csv(bp_results_csv, index=False)

            # ----- MF -----
            print("  Running MF inference...")
            infer_df = pd.read_csv(args.mf_input_csv)
            print(infer_df['label'].value_counts())
            infer_df["Sequence"] = infer_df["Sequence"].str.replace('|'.join(["O", "B", "U", "Z"]), "X", regex=True)
            if needs_spaces:
                infer_df['Sequence'] = infer_df.apply(lambda row: " ".join(row["Sequence"]), axis=1)
            infer_dataset = training_module.CustomProteinDataset(infer_df["Sequence"], infer_df["label"], tokenizer, infer_df["tensor id"])
            infer_loader = DataLoader(infer_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=training_module.collate_fn)

            config = LoraConfig(
                inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["query", "value", "key"]
            )
            MF_model = get_peft_model(model, config)
            training_module.print_trainable_parameters(MF_model)
            MF_model = training_module.CustomProtBERTModel(MF_model, 3)
            training_module.print_trainable_parameters(MF_model)
            MF_model.to(device)

            print(f"    Loading MF weights from {mf_bw_path}")
            MF_model = training_module.load_model(MF_model, mf_bw_path)
            MF_model.to(device)

            MF_infer_results, MF_features, _ = training_module.infer_4_chosun(MF_model, infer_loader, bacteria_results_dir)
            mf_pred_labels = MF_infer_results.drop(columns=['tensor id']).idxmax(axis=1).to_numpy()

            mf_probs = MF_infer_results[[col for col in MF_infer_results.columns if col != 'tensor id']].values.astype(np.float32)
            MF_infer_results.columns = [str(col) for col in MF_infer_results.columns]
            tsne_plot(bacteria_results_dir, MF_features, infer_df['label'].to_numpy(), bacteria_name, "MF", mf_class_names, model_name, args.tsne_perplexities)
            prob_cols = [col for col in MF_infer_results.columns if col != 'tensor id']
            MF_infer_results['MF pred value'] = MF_infer_results[prob_cols].idxmax(axis=1)

            merged_df = pd.merge(infer_df.copy(), MF_infer_results, on='tensor id', how='left')
            true_labels_array = infer_df['label'].values
            merged_df['True Label'] = merged_df['label'].map({i: mf_class_names[i] for i in range(len(mf_class_names))})
            merged_df['MF Misclassified'] = ['Yes' if true_labels_array[i] != mf_pred_labels[i] else 'No' for i in range(len(true_labels_array))]

            cols_to_move = ['MF pred value']
            other_cols = [col for col in merged_df.columns if col not in cols_to_move]
            merged_df = merged_df[cols_to_move + other_cols]
            merged_df.drop(columns=['label'], inplace=True)

            print(f"    MF Predictions: {pd.Series(mf_pred_labels).value_counts().to_dict()}")
            mf_metrics = compute_metrics(infer_df['label'].values, mf_pred_labels, mf_probs, 3)

            # Save MF results CSV (keep tensor id and Sequence for merging)
            mf_results_csv = os.path.join(bacteria_results_dir, f"{bacteria_name}_{model_name}_MF_results.csv")
            merged_df.to_csv(mf_results_csv, index=False)

    # Merge all results for each bacteria
    print(f"\n{'='*80}")
    print("Merging results by bacteria...")
    print(f"{'='*80}")
    merge_results_by_bacteria(args.results_dir_base)
    print("All inference complete!")
