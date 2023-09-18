import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig


class PromptCL(nn.Module):
    def __init__(self, encoder, bert_model, temp, device, num_label=4, dropout=0.5, alpha=0.15, special_tokenizer=None):
        super(PromptCL, self).__init__()
        self.config = AutoConfig.from_pretrained(bert_model)
        # self.model = AutoModel.from_pretrained(bert_model)
        self.model = encoder
        if special_tokenizer is not None:
            self.model.encoder.resize_token_embeddings(len(special_tokenizer))
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        #         self.dense_mark = nn.Linear(self.config.hidden_size*2, self.config.hidden_size*2)
        self.activation = nn.Tanh()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
        self.device = device

        # classification
        self.alpha = alpha
        self.classifier = nn.Linear(self.config.hidden_size * 2, num_label)
        self.dropout = nn.Dropout(dropout)
        self.to(device)

    # def cl_loss(self, pooler_output):
    #     # 得到y_pred对应的label, 每第三句没有label, 跳过, label= [1, 0, 4, 3, ...]
    #     y_true = torch.arange(pooler_output.shape[0]).to(self.device)
    #     use_row = torch.where((y_true + 1) % 3 != 0)[0]
    #     y_true = (use_row - use_row % 3 * 2) + 1
    #     # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    #     sim = F.cosine_similarity(pooler_output.unsqueeze(1), pooler_output.unsqueeze(0), dim=-1)
    #     # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    #     sim = sim - torch.eye(pooler_output.shape[0], device=self.device) * 1e12
    #     # 选取有效的行
    #     sim = torch.index_select(sim, 0, use_row)
    #     # 相似度矩阵除以温度系数
    #     sim = sim / self.temp
    #     # 计算相似度矩阵与y_true的交叉熵损失
    #     loss = F.cross_entropy(sim, y_true)
    #     return loss

    def forward(self, moment, input_ids, attention_mask, token_type_ids=None, ind = None, entity_idx=None, classify_labels=None,
                use_cls=False):
        batch_size = input_ids.size(0)
        num_sent = input_ids.size(1)
        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len) (32*2, 32)


        logits, mask_hidden, cls_hidden = self.model.mask_replay_forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_mask_hidden=True,
            return_cls_hidden=True)

        # class

        logits = logits.view(batch_size, num_sent, -1)[:, 0]
        # classify_labels = torch.cat([classify_labels, classify_labels])

        loss_ce = nn.CrossEntropyLoss()
        ce_loss = loss_ce(logits, classify_labels.view(-1))

        # # cls
        if use_cls:
            hidden = cls_hidden.view((batch_size, num_sent, -1))  # last_hidden
            # pooler_output = hidden
            pooler_output = self.dense(hidden)
            pooler_output = self.activation(pooler_output)
            z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
        # mask
        else:
            hidden = mask_hidden.view((batch_size, num_sent, -1))  # last_hidden
            # pooler_output = hidden
            pooler_output = self.dense(hidden)
            pooler_output = self.activation(pooler_output)
            z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
        # z1 = z1-z3
        # z2 = z2-z4
        cos_sim = self.cos(z1.unsqueeze(1), z2.unsqueeze(0)) / self.temp

        Odis_function = nn.PairwiseDistance(p=2)
        distance = Odis_function(F.normalize(z1, p=2, dim=1).unsqueeze(1), F.normalize(z2, p=2, dim=1).unsqueeze(0))
        mask_combined = torch.eye(classify_labels.shape[0], device=self.device)
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        pos_mask_combined = (classify_labels.unsqueeze(1).repeat(1, classify_labels.shape[0]) == classify_labels).to(self.device)
        neg_mask_combined = 1 - pos_mask_combined.long()

        neg_number = neg_mask_combined.sum(1, keepdim=True)
        neg_weights = neg_number * F.softmax(torch.masked_fill(-distance, pos_mask_combined.bool(), value=-1e7), dim=1) + pos_mask_combined

        exp_dot_tempered = torch.exp(cos_sim)
        log_prob = -torch.log(
            exp_dot_tempered / (torch.sum(exp_dot_tempered * neg_weights, dim=1, keepdim=True)) + 1e-5)
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        cl_loss = torch.mean(supervised_contrastive_loss_per_sample)

        # method2
        # neg_mask_combined = (classify_labels.unsqueeze(1).repeat(1, classify_labels.shape[0]) != classify_labels).to(self.device)  # n*m
        # mask_combined = torch.eye(classify_labels.shape[0], device=self.device)
        # exp_dot_tempered = torch.exp(cos_sim)
        #
        # cardinality_per_samples = torch.sum(mask_combined, dim=1)
        #
        # log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * neg_mask_combined, dim=1, keepdim=True))+ 1e-5)
        # supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        # cl_loss = torch.mean(supervised_contrastive_loss_per_sample)

        # method1
        # con_labels = torch.arange(cos_sim.size(0)).long().to(self.device)
        # loss_fct = nn.CrossEntropyLoss()
        # cl_loss = loss_fct(cos_sim, con_labels)

        return ce_loss+ self.alpha * cl_loss

    # @torch.no_grad()
    # def embed(self, encoder, input_ids, attention_mask, token_type_ids=None, entity_idx=None, use_cls=False):
    #     batch_size = input_ids.size(0)
    #     num_sent = input_ids.size(1)
    #     # Flatten input for encoding
    #     input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
    #     attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
    #     if token_type_ids is not None:
    #         token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len) (32*2, 32)
    #
    #     logits, mask_hidden, cls_hidden = encoder.model.mask_replay_forward(
    #         input_ids=input_ids,
    #         token_type_ids=token_type_ids,
    #         attention_mask=attention_mask,
    #         return_mask_hidden=True,
    #         return_cls_hidden=True)
    #     # # cls
    #     if use_cls:
    #         hidden = cls_hidden.view((batch_size, num_sent, -1))  # last_hidden
    #         pooler_output = encoder.dense(hidden)
    #         pooler_output = encoder.activation(pooler_output)
    #         z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
    #
    #     # mask
    #     else:
    #         hidden = mask_hidden.view((batch_size, num_sent, -1))  # last_hidden
    #         pooler_output = encoder.dense(hidden)
    #         pooler_output = encoder.activation(pooler_output)
    #         z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
    #     return z1

    def encode(self, input_ids, attention_mask, token_type_ids=None, entity_idx=None, use_cls=False):

        logits, mask_hidden, cls_hidden = self.model.mask_replay_forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_mask_hidden=True,
            return_cls_hidden=True)

        if use_cls:
            cls_token = cls_hidden
            pooler_output = self.dense(cls_token)
            sent_emb = self.activation(pooler_output)
            # sent_emb = cls_token
            return sent_emb
        else:
            cls_token = mask_hidden
            pooler_output = self.dense(cls_token)
            sent_emb = self.activation(pooler_output)
            # sent_emb = cls_token
            return sent_emb