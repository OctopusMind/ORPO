import torch
import torch.nn as nn
import torch.nn.functional as F


class ORPO:
    def __init__(self, model, model_opt, config, tokenizer):
        self.model = model
        self.model_opt = model_opt
        self.alpha = config.alpha
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    def train(self, inputs_ids, attention_mask, labels_mask):
        logits = self.model(inputs_ids, attention_mask)
        # 计算sft_loss
        len_chosen = int(inputs_ids.shape[0] / 2)
        sft_loss = self.sft_loss(logits, inputs_ids, len_chosen)
        # 计算odds loss
        policy_token_probs = self.probs_from_logits(logits[:, :-1, :], inputs_ids[:, 1:])
        policy_probs = self.filter_mask(policy_token_probs, labels_mask)  # 公式是论文的(4),这里是生成一个句子一起的概率
        odds_loss = self.orpo_loss(policy_probs, len_chosen)
        # 反向传播
        loss = sft_loss - odds_loss.mean()  # 论文公式6
        self.model_opt.zero_grad()
        loss.backward()
        self.model_opt.step()
        print(loss)

    def orpo_loss(self, policy_probs, len_chosen):
        """
        计算公式
        详细过程可以看博客：
        :param policy_probs: 策略模型的probs
        :return: loss
        """

        def concat_probs(probs, len_chosen):
            """
            拆开合理与不合理数据的probs
            :param probs: 策略模型的probs
            :return: 合理和不合理数据的probs
            """
            rejected_data = torch.cat(probs[:len_chosen])
            chosen_data = torch.cat(probs[len_chosen:])
            return rejected_data, chosen_data

        neg_prob, pos_prob = concat_probs(policy_probs, len_chosen)  # 计算合理数据的probs和不合理数据的probs
        # 下面是论文的公式5，计算两个的比值，这个值越大越好，即生成偏好好的概率大于生成消极的可能性
        # 下面的实现和源码实现不同,因为pos_prob是概率和，没有取log再求和，源码实现：policy_chosen_logps就是pos_prob， policy_rejected_logps就是neg_prob
        # log_odds = (policy_chosen_logps - policy_rejected_logps) -
        # (torch.log1p(-torch.exp(policy_chosen_logps)) - torch.log1p(-torch.exp(policy_rejected_logps)))
        log_odds = (torch.log(pos_prob) - torch.log(neg_prob)) - torch.log(1 - pos_prob) + torch.log(1 - neg_prob)
        # 下面两行是公式7,OR偏好比值先log(防止值过大，防止梯度爆炸)， 在sigmoid到0到1(防止值过大，防止梯度爆炸)
        sig_ratio = torch.nn.functional.sigmoid(log_odds)
        ratio = torch.log(sig_ratio)
        odds_loss = self.alpha * ratio
        print(f"pos_prob:{pos_prob}, neg_prob:{neg_prob}, odds_loss:{odds_loss}")
        return odds_loss

    def sft_loss(self, logits, inputs_ids, len_chosen):
        chosen_logits = logits[:len_chosen, :-1, :].contiguous().view(-1, logits.shape[-1])
        chosen_inputs_ids = inputs_ids[:len_chosen, 1:].contiguous().view(-1)
        sft_loss = self.loss_fct(chosen_logits, chosen_inputs_ids)
        return sft_loss

    @staticmethod
    def probs_from_logits(logits, labels):
        log_probs = F.softmax(logits, dim=2)
        probs = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(-1)
        return probs

    @staticmethod
    def filter_mask(probs, labels_masks):
        """
        :param probs: softmax之后的概率
        :param labels_masks:label 对应的mask
        :return: 去除padding之后的数据
        """
        return [value[one_response_ids_mask[:-1] == 1].sum().unsqueeze(0)/one_response_ids_mask[:-1].sum() for value, one_response_ids_mask in
                zip(probs, labels_masks)]
