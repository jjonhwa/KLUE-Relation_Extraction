import torch
from torch import nn
from torch.cuda.amp import autocast
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from utils.loss import FocalLoss


class IBModel(nn.Module):
    """
    Model implementation of paper: 'An Improved Baseline for Sentence-level Relation Extraction'
    with some customizations added to match the performance on KLUE RE task
    https://arxiv.org/abs/2102.01373 | https://github.com/wzhouad/RE_improved_baseline/blob/main/model.py
    """

    def __init__(self, model_name, config):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size
        self.loss_fnt = FocalLoss(gamma=1.0)
        
        # Encoder의 출력 결과물로 entity_1과 entity_2의 start token에 대한 embedding을 받는다.
        # 두 개의 hidden state vector를 concatenate한 후 FFN을 거친다.
        # FFN -> ReLU를 차례로 수행한 후, 다시 한 번 FFN을 거친 후에 Softmax로 확률값이 도출된다.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), 
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size, config.num_labels),
        )

    @autocast()
    def forward(self, input_ids=None, attention_mask=None, labels=None, ss=None, os=None):
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
        ) # Backbone Model에서의 출력을 받는다.
        
        pooled_output = outputs[0]
        idx = torch.arange(input_ids.size(0)).to(input_ids.device)
        
        ss_emb = pooled_output[idx, ss]         # entity1의 start token에 대한 embedding
        os_emb = pooled_output[idx, os]         # entity2의 start token에 대한 embedding
        
        h = torch.cat((ss_emb, os_emb), dim=-1) # 두 embedding을 concatenate
        logits = self.classifier(h)             # Improved Baseline Model 로직 적용
        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fnt(logits.float(), labels)
            outputs = (loss,) + outputs
        return outputs

class FCLayer(nn.Module):
    """
    R-BERT: https://github.com/monologg/R-BERT/blob/master/model.py
    R-BERT에서 반복적으로 사용되는 Fully Connected Layer를 정의한다.
    """

    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RBERT(nn.Module):
    """
    R-BERT: https://github.com/monologg/R-BERT/blob/master/model.py
    
    R-BERT:
        CLS hidden state vector
        Entity1의 각 Token에 대한 Average hidden state vector
        Entity2의 각 Token에 대한 Average hidden state vector
        
        각 hidden state vector는 Fully-Connected Layer + Activation을 통과한다.
        
        통과한 3개의 Vector를 Concatenate하여 하나의 Vector로 만들어주고 다시 Fully-Connected Layer를 통과한다.
        
        그 후, Softmax를 통과하여 최종 예측을 진행
    """

    def __init__(
        self,
        model_name: str = "klue/roberta-large",
        num_labels: int = 30,
        dropout_rate: float = 0.1,
        special_tokens_dict: dict = None,
        is_train: bool = True,
    ):
        super(RBERT, self).__init__()

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.backbone_model = AutoModel.from_pretrained(model_name, config=config)
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        config.num_labels = num_labels

        # 사전 Dataset에 Typed Entity Marker를 통해 추가된 Special Token을 Tokenizer에 추가한다.
        self.special_tokens_dict = special_tokens_dict
        self.backbone_model.resize_token_embeddings(len(self.tokenizer))

        # CLS Token의 Hidden state vector가 통과할 FCLayer 정의
        self.cls_fc_layer = FCLayer(
            config.hidden_size, config.hidden_size, self.dropout_rate
        )
        
        # Entity1, Entity2의 각 Token에 대한 Average Hidden state vector가 통과할 FCLayer 정의
        # Entity1과 Entity2에서 사용하는 FCLayer는 서로 weight를 공유한다.
        self.entity_fc_layer = FCLayer(
            config.hidden_size, config.hidden_size, self.dropout_rate
        )
        
        # 최종, CLS, Entity1, Entity2가 Concatenate된 vector가 통과할 FCLayer 정의
        self.label_classifier = FCLayer(
            config.hidden_size * 3,
            self.num_labels,
            self.dropout_rate,
            use_activation=False,
        )

    def entity_average(self, hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(
        self,
        input_ids,
        attention_mask,
        subject_mask=None,
        object_mask=None,
        labels=None,
    ):

        outputs = self.backbone_model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        sequence_output = outputs["last_hidden_state"]
        pooled_output = outputs[
            "pooler_output"
        ]  # [CLS] token's hidden featrues(hidden state)

        # hidden state's average in between entities
        # print(sequence_output.shape, subject_mask.shape)
        e1_h = self.entity_average(sequence_output, subject_mask)  # token in between subject entities ->
        e2_h = self.entity_average(sequence_output, object_mask)  # token in between object entities

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)  # [CLS] token -> hidden state | green on diagram
        e1_h = self.entity_fc_layer(e1_h)  # subject entity's fully connected layer | yellow on diagram
        e2_h = self.entity_fc_layer(e2_h)  # object entity's fully connected layer | red on diagram

        # Concat -> fc_layer / [CLS], subject_average, object_average
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)
        return logits
