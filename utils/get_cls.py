import torch

from transformers import AutoTokenizer, AutoModel

def get_model_tokenizer(model_name) :
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return model, tokenizer

def get_cls_token(sent_A, model, tokenizer):
    model.eval()
    tokenized_sent = tokenizer(
            sent_A,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
            max_length=128
    )
    with torch.no_grad():# 그라디엔트 계산 비활성화
        outputs = model(    # **tokenized_sent
            input_ids=tokenized_sent['input_ids'],
            attention_mask=tokenized_sent['attention_mask'],
            token_type_ids=tokenized_sent['token_type_ids']
            )
    logits = outputs.last_hidden_state[:,0,:].detach().cpu().numpy()
    return logits

def get_dataset_cls_hidden(question: list) :
    dataset_cls_hidden = []
    for q in question :
        q_cls = get_cls_token(q)
        dataset_cls_hidden.append(q_cls)

    dataset_cls_hidden = np.array(dataset_cls_hidden).squeeze(axis = 1)
    return dataset_cls_hidden