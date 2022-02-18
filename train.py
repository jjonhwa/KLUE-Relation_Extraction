# import python innate modules
import argparse
import json
import random
import os

# import data wrangling modules
import pandas as pd
import numpy as np

# import machine learning modules
from sklearn.metrics import f1_score, confusion_matrix

# import torch and its applications
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

# import transformers and its applications
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AdamW,
    get_cosine_schedule_with_warmup,
)
from transformers.optimization import get_linear_schedule_with_warmup

# import third party modules
import yaml
from tqdm import tqdm
from easydict import EasyDict
from adamp import AdamP

# import custom modules
from dataset import *
from models import *
from utils.metrics import *

DATA_CFG = {}
IB_CFG = {}
RBERT_CFG = {}
CONCAT_CFG = {}

# Read config.yaml file
with open("config.yaml") as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)

DATA_CFG = SAVED_CFG["data"]
IB_CFG = SAVED_CFG["IB"]
RBERT_CFG = SAVED_CFG["RBERT"]
CONCAT_CFG = SAVED_CFG["Concat"]


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministric = True
    torch.backends.cudnn.benchmark = True


def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def train_concat():
    data_dir = DATA_CFG.pororo_train_path
    dataset = pd.read_csv(data_dir)
    model_name = CONCAT_CFG.pretrained_model_name
    special_token_path = DATA_CFG.pororo_special_token_path

    training_args = TrainingArguments(
        output_dir="./results",  # output directory
        save_total_limit=5,  # number of total save model.
        save_steps=CONCAT_CFG.save_steps,  # model saving step.
        num_train_epochs=CONCAT_CFG.num_train_epochs,  # total number of training epochs
        learning_rate=CONCAT_CFG.learning_rate,  # learning_rate
        per_device_train_batch_size=CONCAT_CFG.batch_size,  # batch size per device during training
        per_device_eval_batch_size=32,  # batch size for evaluation
        warmup_steps=CONCAT_CFG.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=CONCAT_CFG.weight_decay,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=100,  # log saving step.
        evaluation_strategy="steps",  # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=500,  # evaluation step.
        load_best_model_at_end=True,
    )

    special_token_list = []
    with open(special_token_path, "r", encoding="UTF-8") as f:
        for token in f:
            special_token_list.append(token.split("\n")[0])

    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
    added_token_num = tokenizer.add_special_tokens(
        {"additional_special_tokens": list(set(special_token_list))}
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    stf = StratifiedKFold(
        n_splits=CONCAT_CFG.num_folds, shuffle=True, random_state=seed_everything(42)
    )
    for fold, (train_idx, dev_idx) in enumerate(
        stf.split(dataset, list(dataset["label"]))
    ):
        print("Fold {}".format(fold + 1))
        model_config = AutoConfig.from_pretrained(CONCAT_CFG.pretrained_model_name)
        model_config.num_labels = 30

        model = AutoModelForSequenceClassification.from_pretrained(
            CONCAT_CFG.pretrained_model_name, config=model_config
        )
        model.to(device)

        # 추가한 token 개수만큼 token embedding size 늘려주기
        model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)

        train_dataset = dataset.iloc[train_idx]
        dev_dataset = dataset.iloc[dev_idx]

        train_label = label_to_num(train_dataset["label"].values)
        dev_label = label_to_num(dev_dataset["label"].values)

        tokenized_train = tokenized_dataset_concat(
            train_dataset, tokenizer, CFG.max_token_length
        )
        tokenized_dev = tokenized_dataset_concat(
            dev_dataset, tokenizer, CFG.max_token_length
        )

        RE_train_dataset = RE_Dataset(tokenized_train, train_label)
        RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=RE_train_dataset,  # training dataset
            eval_dataset=RE_dev_dataset,  # evaluation dataset
            compute_metrics=compute_metrics,  # define metrics function
        )
        trainer.train()

        makedirs(f"./best_model/fold_{fold}")
        model.save_pretrained(f"./best_model/fold_{fold}/")

        model.cpu()
        del model
        torch.cuda.empty_cache()


def train_ib():

    data_dir = DATA_CFG.train_file_path
    kfold_num = IB_CFG.num_folds
    dataset = load_data(data_dir)
    kfold_dataset = split_df(dataset, kfold_n=kfold_num)

    ## Save best model on validation for each fold ##
    for fold_num in range(kfold_num):
        MODEL_NAME = IB_CFG.pretrained_model_name
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        train_dataset = kfold_dataset[fold_num][0]
        valid_dataset = kfold_dataset[fold_num][1]

        ### Data Loader ###
        train_features = processor(tokenizer, train_dataset, train_mode=True)
        val_features = processor(tokenizer, valid_dataset, train_mode=True)
        train_dataloader = DataLoader(
            train_features,
            batch_size=IB_CFG.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(device)
        model_config = AutoConfig.from_pretrained(MODEL_NAME)
        model_config.num_labels = 30
        model = IBModel(MODEL_NAME, config=model_config)
        model.parameters
        model.to(device)

        total_steps = (
            int(len(train_dataloader) * IB_CFG.num_train_epochs)
            // IB_CFG.gradient_accumulation_steps
        )
        warmup_steps = int(total_steps * IB_CFG.warmup_ratio)
        scaler = GradScaler()

        optimizer = AdamP(
            model.parameters(),
            lr=IB_CFG.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        
        num_steps = 0
        best_f1 = 0
        print(f"Fold num: {fold_num}/5")
        for epoch in range(int(IB_CFG.num_train_epochs)):
            print(f"Epoch num: {epoch}")
            model.zero_grad()
            average_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                model.train()
                inputs = {
                    "input_ids": batch[0].to(device),
                    "attention_mask": batch[1].to(device),
                    "labels": batch[2].to(device),
                    "ss": batch[3].to(device),
                    "os": batch[5].to(device),
                }
                outputs = model(**inputs)
                loss = outputs[0] / IB_CFG.gradient_accumulation_steps
                average_loss += loss
                scaler.scale(loss).backward()
                if step % IB_CFG.gradient_accumulation_steps == 0:
                    num_steps += 1
                    if IB_CFG.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), IB_CFG.max_grad_norm
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    model.zero_grad()
                    
            print(f"average_training_loss: {average_loss/len(train_dataloader)}")
            f1, auprc, acc = evaluate_ib(model, val_features)
            if f1 > best_f1:
                best_f1 = f1
                torch.save(
                    model,
                    f"./{DATA_CFG.result_dir}/roberta_focal_adamp{fold_num}.pt",
                )


def evaluate_ib(model, features):
    dataloader = DataLoader(
        features, batch_size=5, collate_fn=collate_fn, drop_last=False
    )
    keys, preds, pred_logitss = [], [], []
    device = torch.device("cuda")
    for i_b, batch in enumerate(dataloader):
        model.eval()
        inputs = {
            "input_ids": batch[0].to(device),
            "attention_mask": batch[1].to(device),
            "ss": batch[3].to(device),
            "os": batch[5].to(device),
        }
        keys += batch[2].tolist()
        with torch.no_grad():
            logit = model(**inputs)[0]
            pred = torch.argmax(logit, dim=-1)
            for i in logit:
                pred_logitss.append(i.tolist())
        preds += pred.tolist()

    keys = np.array(keys, dtype=np.int64)
    f1, auprc, acc = compute_metrics(keys, pred_logitss)
    output = {"f1": f1, "auprc": auprc, "acc": acc}
    print(output)
    return f1, auprc, acc


def train_rbert():
    # read pororo dataset
    df_pororo_dataset = pd.read_csv(DATA_CFG.pororo_train_path)

    # remove the first index column
    df_pororo_dataset = df_pororo_dataset.drop(df_pororo_dataset.columns[0], axis=1)

    # fetch tokenizer
    tokenizer = AutoTokenizer.from_pretrained(RBERT_CFG.model_name)

    # fetch special tokens annotated with ner task
    special_token_list = []
    with open(DATA_CFG.pororo_special_token_path, "r", encoding="UTF-8") as f:
        for token in f:
            special_token_list.append(token.split("\n")[0])

    if torch.cuda.is_available() and RBERT_CFG.debug == False:
        device = torch.device("cuda")
        print(f"There are {torch.cuda.device_count()} GPU(s) available.")
        print("GPU Name:", torch.cuda.get_device_name(0))
    else:
        print("No GPU, using CPU.")
        device = torch.device("cpu")

    criterion = FocalLoss(gamma=RBERT_CFG.gamma)  # 0.0 equals to CrossEntropy

    train_data = RBERT_Dataset(df_pororo_dataset, tokenizer)
    dev_data = RBERT_Dataset(df_pororo_dataset, tokenizer)

    stf = StratifiedKFold(
        n_splits=RBERT_CFG.num_folds, shuffle=True, random_state=seed_everything(42)
    )

    for fold_num, (train_idx, dev_idx) in enumerate(
        stf.split(df_pororo_dataset, list(df_pororo_dataset["label"]))
    ):

        print(f"#################### Fold: {fold_num + 1} ######################")

        train_set = Subset(train_data, train_idx)
        dev_set = Subset(dev_data, dev_idx)

        train_loader = DataLoader(
            train_set,
            batch_size=RBERT_CFG.batch_size,
            shuffle=True,
            num_workers=RBERT_CFG.num_workers,
        )
        dev_loader = DataLoader(
            dev_set,
            batch_size=RBERT_CFG.batch_size,
            shuffle=False,
            num_workers=RBERT_CFG.num_workers,
        )

        # fetch model
        model = RBERT(RBERT_CFG.model_name)
        model.to(device)

        # fetch loss function, optimizer, scheduler outside of torch library
        # https://github.com/clovaai/AdamP
        optimizer = AdamP(
            model.parameters(),
            lr=RBERT_CFG.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=RBERT_CFG.weight_decay,
        )
        # optimizer = AdamW(model.parameters(), lr=RBERT_CFG.learning_rate, betas=(0.9, 0.999), weight_decay=CFG.weight_decay) # AdamP is better

        # https://huggingface.co/transformers/main_classes/optimizer_schedules.html
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=RBERT_CFG.warmup_steps,
            num_training_steps=len(train_loader) * RBERT_CFG.num_epochs,
        )

        best_eval_loss = 1.0
        steps = 0

        # fetch training loop
        for epoch in tqdm(range(RBERT_CFG.num_epochs)):
            train_loss = Metrics()
            dev_loss = Metrics()
            for _, batch in enumerate(train_loader):
                optimizer.zero_grad()

                label = batch["label"].to(device)
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "subject_mask": batch["subject_mask"].to(device),
                    # 'token_type_ids' # NOT FOR ROBERTA!
                    "object_mask": batch["object_mask"].to(device),
                    "label": label,
                }

                # model to training mode
                model.train()
                pred_logits = model(**inputs)
                loss = criterion(pred_logits, label)

                # backward
                loss.backward()
                optimizer.step()
                scheduler.step()

                # update metrics
                train_loss.update(loss.item(), len(label))

                steps += 1
                # for every 100 steps
                if steps % 100 == 0:
                    print(
                        "Epoch: {}/{}".format(epoch + 1, RBERT_CFG.num_epochs),
                        "Step: {}".format(steps),
                        "Train Loss: {:.4f}".format(train_loss.avg),
                    )
                    for dev_batch in dev_loader:
                        dev_label = dev_batch["label"].to(device)
                        dev_inputs = {
                            "input_ids": dev_batch["input_ids"].to(device),
                            "attention_mask": dev_batch["attention_mask"].to(device),
                            "subject_mask": dev_batch["subject_mask"].to(device),
                            # 'token_type_ids' # NOT FOR ROBERTA!
                            "object_mask": dev_batch["object_mask"].to(device),
                            "label": dev_label,
                        }

                        # switch model to eval mode
                        model.eval()
                        dev_pred_logits = model(**dev_inputs)
                        loss = criterion(dev_pred_logits, dev_label)

                        # update metrics
                        dev_loss.update(loss.item(), len(dev_label))

                    # print metrics
                    print(
                        "Epoch: {}/{}".format(epoch + 1, RBERT_CFG.num_epochs),
                        "Step: {}".format(steps),
                        "Dev Loss: {:.4f}".format(dev_loss.avg),
                    )

                    if best_eval_loss > dev_loss.avg:
                        best_eval_loss = dev_loss.avg
                        torch.save(
                            model.state_dict(),
                            "./results/{}-fold-{}-best-eval-loss-model.pt".format(
                                fold_num + 1, RBERT_CFG.num_folds
                            ),
                        )
                        print(
                            "Saved model with lowest validation loss: {:.4f}".format(
                                best_eval_loss
                            )
                        )
                        early_stop = 0
                    else:
                        early_stop += 1
                        if early_stop > 2:
                            break

        # Prevent OOM error
        model.cpu()
        del model
        torch.cuda.empty_cache()
        
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='custom', help='custom(custom concat) or rbert or ib(Improved Baseline)')
    args = parser.parse_args()
    
    if args.mode == 'custom':
        train_concat()
    elif args.mode == 'rbert':
        train_rbert()
    elif args.mode == 'ib':
        train_ib()
    else:
        raise ValueError("Inappropriate values have been received.")
    
    
