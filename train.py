import pandas as pd
import numpy as np
import os
import argparse
import math
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.optim import AdamW, lr_scheduler
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassMatthewsCorrCoef, MulticlassAUROC
import warnings
import datetime
from transformers import AutoModel, AutoTokenizer


# Globals to be set by the main script
tokenizer = None
device = None
args = None
main_output_save_folder_path = None

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
class CustomProteinDataset(Dataset):
    def __init__(self, sequence, label, tokenizer, id):
        self.sequence = sequence
        self.label = label
        self.tokenizer = tokenizer
        self.id = id

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        return {"sequence": self.sequence[idx], "labels": self.label[idx], "ids": self.id[idx]}

def collate_fn(data):
    embs = [d['sequence'] for d in data]
    tokens = tokenizer(embs, padding='longest', return_tensors='pt', max_length=1024, truncation=True) 
    ids = torch.tensor([d['ids'] for d in data])
    labels = torch.tensor([d['labels'] for d in data])
    # encodings = tokenizer(text = tjdgs["attention_mask"])
    return {"input_ids": tokens.input_ids, "attention_mask": tokens.attention_mask, "labels": labels, "ids": ids, "seqs": embs}

class CustomProteinDataset_inference(Dataset):
    def __init__(self, sequence, label, tokenizer, id):
        self.sequence = sequence
        self.label = label
        self.tokenizer = tokenizer
        self.id = id

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        return {"sequence": self.sequence[idx], "labels": self.label[idx], "ids": self.id[idx]}

class CustomProtBERTModel(nn.Module):
    def __init__(self, model, num_labels):
        super(CustomProtBERTModel, self).__init__()

        self.model = model
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.2)
        # determine hidden size for different model configs (BERT / ESM2 / others)
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.model.config, "embed_dim", None)
        if hidden_size is None:
            hidden_size = getattr(self.model.config, "d_model", None)
        if hidden_size is None:
            raise ValueError("Could not determine hidden size from model.config")
        self.hidden_size = hidden_size
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.final_proj = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self,  input_ids=None, attention_mask=None, return_attentions=None):
        # Call underlying model; allow different return objects from various HF implementations
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=return_attentions)
        # prefer last_hidden_state if present
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            # some models may return 'hidden_states' or have different structure; try common alternatives
            last_hidden = outputs[0]
        x = torch.mean(last_hidden, dim=1)
        final_hidden_embedding = self.dropout(x)
        logits = self.final_proj(final_hidden_embedding)
        attentions = getattr(outputs, "attentions", None)
        # CrossEntropyLoss expects unnormalized logits.  Call softmax only at
        # inference/export boundaries where class probabilities are required.
        return logits, final_hidden_embedding, attentions
   
def train(model, train_loader, optim, criterion, epoch, scheduler):
    model.train()

    losses = []
    count = 0
    # LABELS = torch.tensor(()).to(device).int()
    # PREDS = torch.tensor(()).to(device)
    # IDS = torch.tensor(()).to(device)
    all_labels = []
    all_preds = []
    all_ids = []
    for batch in tqdm(train_loader, desc="Train"):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        ids = batch["ids"].to(device)
        logits, final_features, attentions = model(input_ids, attention_mask, None)
        # _, preds = torch.max(logits, 1)

        # compute loss
        loss = criterion(logits, labels)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        count += 1
        all_labels.append(labels.detach().cpu())
        all_preds.append(logits.detach().cpu())
        all_ids.append(ids.detach().cpu())
    LABELS = torch.cat(all_labels)
    PREDS = torch.cat(all_preds)
    IDS = torch.cat((all_ids))
    
    # testing to find error
    torch.cuda.empty_cache()
    ###
    scheduler.step()
    print("lr: ", optim.param_groups[0]['lr'],"\n")
    acc, f1, mcc, auroc = eval_metrics(LABELS, PREDS, args.n_classes)

    avg_loss = np.mean(losses)
    avg_acc = acc.item()
    avg_f1 = f1.item()
    avg_mcc = mcc.item()
    avg_auroc = auroc.item()

    return [avg_loss, avg_acc, avg_f1, avg_mcc, avg_auroc]

def eval(model, valid_loader, criterion, epoch, type):
    model.eval()
    losses = []
    count = 0
    # LABELS = torch.tensor(()).to(device).int()
    # PREDS = torch.tensor(()).to(device)
    # IDS = torch.tensor(()).to(device)
    all_labels = []
    all_preds = []
    all_ids = []
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Valid"):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            ids = batch["ids"].to(device)
            logits, final_features, attentions = model(input_ids, attention_mask, None)

            if count == 0:
                total_final_features = final_features
                total_labels = labels

            else:
                total_final_features = torch.cat([total_final_features, final_features], dim=0)
                total_labels = torch.cat((total_labels, labels), dim=0)


            count += 1
            # LABELS = torch.cat((LABELS, labels))
            # PREDS = torch.cat((PREDS, logits))
            # IDS = torch.cat((IDS, ids))
            # compute loss
            loss = criterion(logits, labels)
            # get metrics
            losses.append(loss.item())
            all_labels.append(labels.detach().cpu())
            all_preds.append(torch.softmax(logits, dim=1).detach().cpu())
            all_ids.append(ids.detach().cpu())
        LABELS = torch.cat(all_labels)
        PREDS = torch.cat(all_preds)
        IDS = torch.cat((all_ids))
        
    # plot output
    tsne_plot(total_final_features.cpu().numpy(), total_labels.cpu().numpy(), type, epoch)

    acc, f1, mcc, auroc = eval_metrics(LABELS, PREDS, args.n_classes)

    # average scores
    avg_loss = np.mean(losses)
    avg_acc = acc.item()
    avg_f1 = f1.item()
    avg_mcc = mcc.item()
    avg_auroc = auroc.item()

    return [avg_loss, avg_acc, avg_f1, avg_mcc, avg_auroc]

def infer_4_chosun(model, infer_loader, save_pth):
    model.eval()
    count = 0
    # LABELS = torch.tensor(()).to(device).int()
    # PREDS = torch.tensor(()).to(device)
    # IDS = torch.tensor(()).to(device)
    all_labels = []
    all_preds = []
    all_ids = []
    with torch.no_grad():
        for batch in tqdm(infer_loader, desc="Inference"):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            ids = batch["ids"].to(device)

            logits, final_features, attentions = model(input_ids, attention_mask, None)

            if count == 0:
                total_final_features = final_features
                total_labels = labels

            else:
                total_final_features = torch.cat([total_final_features, final_features], dim=0)
                total_labels = torch.cat((total_labels, labels), dim=0)

            count += 1

            all_labels.append(labels.detach().cpu())
            all_preds.append(torch.softmax(logits, dim=1).detach().cpu())
            all_ids.append(ids.detach().cpu())
        LABELS = torch.cat(all_labels)
        PREDS = torch.cat(all_preds)
        IDS = torch.cat((all_ids))

    # plot output
    # tsne_plot_inference(total_final_features.cpu().numpy(), total_labels.cpu().numpy(), save_pth)

    df = pd.DataFrame(IDS.cpu().numpy(), columns=["tensor id"]).astype('int64')
    df2 = pd.DataFrame(PREDS.cpu().numpy())
    df3 = pd.concat([df, df2], axis=1)
    
    return df3, total_final_features.cpu().numpy(), total_labels.cpu().numpy()



def infer(model, infer_loader, save_pth):
    model.eval()
    count = 0
    # LABELS = torch.tensor(()).to(device).int()
    # PREDS = torch.tensor(()).to(device)
    # IDS = torch.tensor(()).to(device)
    all_labels = []
    all_preds = []
    all_ids = []
    with torch.no_grad():
        for batch in tqdm(infer_loader, desc="Inference"):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            ids = batch["ids"].to(device)

            logits, final_features, attentions = model(input_ids, attention_mask, None)

            if count == 0:
                total_final_features = final_features
                total_labels = labels

            else:
                total_final_features = torch.cat([total_final_features, final_features], dim=0)
                total_labels = torch.cat((total_labels, labels), dim=0)

            count += 1

            all_labels.append(labels.detach().cpu())
            all_preds.append(logits.detach().cpu())
            all_ids.append(ids.detach().cpu())
        LABELS = torch.cat(all_labels)
        PREDS = torch.cat(all_preds)
        IDS = torch.cat((all_ids))

    # plot output
    # tsne_plot_inference(total_final_features.cpu().numpy(), total_labels.cpu().numpy(), save_pth)

    df = pd.DataFrame(IDS.cpu().numpy(), columns=["tensor id"]).astype('int64')
    df2 = pd.DataFrame(PREDS.cpu().numpy())
    df3 = pd.concat([df, df2], axis=1)
    print(df3)
    
    return df3
def infer_attention_map(model, infer_loader, save_pth):
    model.eval()
    count = 0
    LABELS = torch.tensor(()).to(device).int()
    PREDS = torch.tensor(()).to(device)
    IDS = torch.tensor(()).to(device)

    with torch.no_grad():
        for batch in tqdm(infer_loader, desc="Inference"):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            ids = batch["ids"].to(device)
            seqs = batch['seqs']

            logits, final_features, attentions = model(input_ids, attention_mask, True)

            if count == 0:
                total_final_features = final_features
                total_labels = labels
                total_attentions = attentions
                total_seqs = seqs
                

            else:
                total_final_features = torch.cat([total_final_features, final_features], dim=0)
                total_labels = torch.cat((total_labels, labels), dim=0)
                total_attentions = np.append(total_attentions,attentions)
                total_seqs = np.concatenate((total_seqs, seqs), axis=0)

            count += 1


            LABELS = torch.cat((LABELS, labels))
            PREDS = torch.cat((PREDS,logits))
            IDS = torch.cat((IDS, ids))


    return total_attentions, total_seqs

def eval_metrics(y_true, y_pred, n_classes):
    # print(f"y_true shape: {y_true.shape}, dtype: {y_true.dtype}")
    # print(f"y_pred shape: {y_pred.shape}, dtype: {y_pred.dtype}")
    
    # Check for NaN or Inf values
    if torch.isnan(y_true).any() or torch.isinf(y_true).any():
        print("Warning: y_true contains NaN or Inf values")
    if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
        print("Warning: y_pred contains NaN or Inf values")
    y_true = y_true.cpu().to(device)
    y_pred = y_pred.cpu().to(device)
    acc_metric = MulticlassAccuracy(num_classes=n_classes).to(device)
    F1_metric = MulticlassF1Score(num_classes=n_classes).to(device)
    mcc_metric = MulticlassMatthewsCorrCoef(num_classes=n_classes).to(device)
    auroc_metric = MulticlassAUROC(num_classes=n_classes, thresholds=None).to(device)
    # y_true = torch.flatten(y_true)
    acc = acc_metric(y_pred, y_true)
    f1 = F1_metric(y_pred, y_true)
    mcc = mcc_metric(y_pred, y_true)
    auroc = auroc_metric(y_pred, y_true)
    return acc, f1, mcc, auroc

# Function for saving the model
def save_model(model, model_path):
    path = model_path
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)
    print("model saved!!!")
    print("model saved at: ", path)

def load_model(model, model_path):
    PATH = model_path
    model.load_state_dict(torch.load(PATH), strict=False)
    return model 

def tsne_plot(features, true_labels, type, epoch):
    torch.cuda.empty_cache()
    save_path = main_output_save_folder_path+"results_visualization/"+type+str(epoch)+".png"
    tsne_result = TSNE(n_components=2, perplexity=10).fit_transform(features)
    colormap = plt.cm.gist_ncar
    colorst = [colormap(i) for i in np.linspace(0, 0.9,len(true_labels))] 
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=true_labels)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("t-SNE Visualization of Classification Results")
    plt.savefig(save_path)

def tsne_plot_inference(features, true_labels, save_pth):
    torch.cuda.empty_cache()
    perplexities = [100, 150, 200]
    for i in perplexities:
        tsne_result = TSNE(n_components=2, perplexity=i).fit_transform(features)
        save_path = save_pth+"inference_perplexity_"+str(i)+".png"

        plt.figure(figsize=(10, 6))
        
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c = true_labels)
        plt.legend(*scatter.legend_elements(), title="Classes")
        plt.title("t-SNE Visualization of Classification Results perplexity = "+str(i))
        plt.savefig(save_path)

def save_in_batches(df, path, batch_size=10000):
    if df.empty:
        return

    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    writer = None
    try:
        num_batches = math.ceil(len(df) / batch_size)
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            batch_df = df.iloc[start:end]

            if batch_df.empty:
                continue

            table = pa.Table.from_pandas(batch_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
    finally:
        if writer:
            writer.close()

def save_in_batches(df, path, batch_size=10000):
    num_batches = (len(df) + batch_size - 1) // batch_size
    writer = None
    try:
        for i in tqdm(range(num_batches)):
            start = i * batch_size
            end = (i + 1) * batch_size
            batch_df = df.iloc[start:end]

            # Convert pandas DataFrame → Arrow Table
            table = pa.Table.from_pandas(batch_df, preserve_index=False)

            if writer is None:
                # Create the writer only once with the schema
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
