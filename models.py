

from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
import torch.nn as nn
import torch





class BertForQuestionAnswering(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.loss_fct = nn.CrossEntropyLoss()
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

    def loss(self, start_logits, end_logits, start_positions, end_positions):

        start_loss = self.loss_fct(start_logits, start_positions)
        _, start_pred = torch.max(start_logits, 1)
        start_correct = (start_pred == start_positions).sum().item()

        end_loss = self.loss_fct(end_logits, end_positions)
        _, end_pred = torch.max(end_logits, 1)
        end_correct = (end_pred == end_positions).sum().item()

        total = start_positions.size(0) + end_positions.size(0)
        total_loss = (start_loss + end_loss) / 2
        total_correct = start_correct+end_correct

        return total_loss, total, total_correct




