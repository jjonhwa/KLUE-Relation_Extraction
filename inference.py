from transformers import AutoTokenizer, AutoModelForSequenceClassification

# import dataset wrangler
import numpy as np
import pandas as pd

import os
import yaml
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F

from dataset import *
from utils.metrics import *
from models import *
from dataset import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

def num_to_label(label):
    origin_label = []
    with open("data/dict_label_to_num.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
    new_dict = {value: key for key, value in dict_num_to_label.items()}
    for v in label:
        origin_label.append(new_dict[v])
    return origin_label


def inference_for_ib(model, test_features, device):
    dataloader = DataLoader(
        test_features, batch_size=16, shuffle=False, collate_fn=collate_fn
    )
    model.eval()
    output_pred = []
    output_prob = []
    for step, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0].to(device),
                "attention_mask": batch[1].to(device),
                "ss": batch[3].to(device),
                "es": batch[5].to(device),
            }
            outputs = model(**inputs)
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)
        output_pred.append(result)
        output_prob.append(prob)
    return (
        np.concatenate(output_pred).tolist(),
        np.concatenate(output_prob, axis=0).tolist(),
    )


def load_test_dataset_for_ib(dataset_dir, tokenizer):
    test_dataset = load_data(dataset_dir)
    test_features = processor(tokenizer, test_dataset, train_mode=False)
    return test_dataset["id"], test_features


def inference_ib():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Tokenizer_NAME = IB_CFG["pretrained_model_name"]
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)
    for fold_num in range(5):
        MODEL_NAME = f"./re_finetuned/fold_ensemble/roberta_focal_adamp{fold_num}.pt'"
        model = torch.load(MODEL_NAME)
        model.parameters
        model.to(device)
        test_dataset_dir = DATA_CFG["test_file_path"]
        test_id, test_features = load_test_dataset_for_ib(test_dataset_dir, tokenizer)
        pred_answer, output_prob = inference_for_ib(model, test_features, device)
        pred_answer = num_to_label(pred_answer)
        output = pd.DataFrame(
            {
                "id": test_id,
                "pred_label": pred_answer,
                "probs": output_prob,
            }
        )
        output.to_csv(f"./prediction/to_ensemble/output_p{fold_num}.csv", index=False)
    print("---- Finished making result files for each fold! ----")

    files = os.listdir("./prediction/to_ensemble")

    to_ensemble = [i for i in files if i.endswith(".csv")]
    total = []

    for i in tqdm(to_ensemble):
        df = pd.read_csv("./prediction/to_ensemble/" + i)
        tmp = [literal_eval(df.iloc[i]["probs"]) for i in range(len(df))]
        total.append(tmp)

    avr_total = torch.sum(torch.tensor(total), dim=0) / 5

    result = np.argmax(avr_total, axis=-1)
    pred_answer = result.tolist()
    predsss = num_to_label(pred_answer)

    avr_total = avr_total.tolist()
    test_file = DATA_CFG["test_file_path"]
    test_ids = test_file["id"].tolist()
    output = pd.DataFrame(
        {"id": test_ids, "pred_label": predsss, "probs": avr_total},
    )
    output.to_csv("./prediction/ib_output.csv", index=False)
    print("---- Finished creating Final ensembled file for all folds! ----")


def inference_rbert():

    PORORO_TEST_PATH = DATA_CFG["pororo_test_path"]
    test_dataset = pd.read_csv(PORORO_TEST_PATH)
    # test_dataset = test_dataset.drop(test_dataset.columns[0], axis=1)
    test_dataset["label"] = 100
    print(len(test_dataset))
    MODEL_NAME = RBERT_CFG["pretrained_model_name"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    special_token_list = []
    with open(DATA_CFG["pororo_special_token_path"], 'r', encoding = 'UTF-8') as f :
        for token in f :
            special_token_list.append(token.split('\n')[0])

    added_token_num = tokenizer.add_special_tokens({"additional_special_tokens":list(set(special_token_list))})    
    test_set = RBERT_Dataset(test_dataset, tokenizer, is_training=False)
    print(len(test_set))
    test_data_loader = DataLoader(
        test_set, batch_size=RBERT_CFG["batch_size"], num_workers=RBERT_CFG["num_workers"], shuffle=False
    )
    oof_pred = []  # out of fold prediction list
    for i in range(5):
        model_path = "/opt/ml/klue-level2-nlp-15/notebooks/results/{}-fold-5-best-eval-loss-model.pt".format(
            i + 1
        )
        model = RBERT(RBERT_CFG["pretrained_model_name"], dropout_rate=RBERT_CFG["dropout_rate"])
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()

        output_pred = []
        for i, data in enumerate(tqdm(test_data_loader)):
            with torch.no_grad():
                outputs = model(
                    input_ids=data["input_ids"].to(device),
                    attention_mask=data["attention_mask"].to(device),
                    subject_mask=data["subject_mask"].to(device),
                    object_mask=data["object_mask"].to(device),
                    # token_type_ids=data['token_type_ids'].to(device) # RoBERTa does not use token_type_ids.
                )
            output_pred.extend(outputs.cpu().detach().numpy())
        output_pred = F.softmax(torch.Tensor(output_pred), dim=1)
        oof_pred.append(np.array(output_pred)[:, np.newaxis])

        # Prevent OOM error
        model.cpu()
        del model
        torch.cuda.empty_cache()

    models_prob = np.mean(
        np.concatenate(oof_pred, axis=2), axis=2
    )  # probability of each class
    result = np.argmax(models_prob, axis=-1)  # label
    # print(result, type(result))
    # print(models_prob.shape, list_prob)

    list_prob = models_prob.tolist()

    test_id = test_dataset["id"]
    df_submission = pd.DataFrame(
        {
            "id": test_id,
            "pred_label": num_to_label(result),
            "probs": list_prob,
        }
    )
    df_submission.to_csv("./prediction/submission_RBERT.csv", index=False)


def inference_concat():
    MODEL_NAME = CONCAT_CFG["pretrained_model_name"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    PORORO_TEST_PATH = DATA_CFG["pororo_test_path"]
    test_dataset = pd.read_csv(PORORO_TEST_PATH)
    test_dataset["label"] = 100
    test_label = list(map(int, test_dataset["label"].values))
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    test_id = test_dataset["id"]
    Re_test_dataset = RE_Dataset_concat(tokenized_test, test_label)

    dataloader = DataLoader(Re_test_dataset, batch_size=32, shuffle=False)
    special_token_list = []
    with open(
        DATA_CFG["pororo_special_token_path"],
        "r",
        encoding="UTF-8",
    ) as f:
        for token in f:
            special_token_list.append(token.split("\n")[0])

    # ./best_model/fold_{fold}
    oof_pred = None
    oof_pred = None
    for i in range(5):
        model_name = DATA_CFG["saved_model_dir"] + f"/fold_{i}"
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        added_token_num = tokenizer.add_special_tokens(
            {"additional_special_tokens": list(set(special_token_list))}
        )
        model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        output_pred = []
        for i, data in enumerate(tqdm(dataloader)):
            with torch.no_grad():
                outputs = model(
                    input_ids=data["input_ids"].to(device),
                    attention_mask=data["attention_mask"].to(device),
                    token_type_ids=data["token_type_ids"].to(device),
                )
            logits = outputs[0]
            prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
            output_pred.append(prob)
        final_prob = np.concatenate(output_pred, axis=0)

        if oof_pred is None:
            oof_pred = final_prob / 5
        else:
            oof_pred += final_prob / 5

    result = np.argmax(oof_pred, axis=-1)
    pred_answer = num_to_label(result)
    output_prob = oof_pred.tolist()

    output = pd.DataFrame(
        {
            "id": test_id,
            "pred_label": pred_answer,
            "probs": output_prob,
        }
    )
    output.to_csv(
        "./prediction/submission_concat.csv", index=False
    )  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='custom', help='custom(custom concat) or rbert or ib(Improved Baseline)')
    args = parser.parse_args()
    
    if args.mode == 'custom':
        inference_concat()
    elif args.mode == 'rbert':
        inference_rbert()
    elif args.mode == 'ib':
        inference_ib()
    else:
        raise ValueError("Inappropriate values have been received.")
