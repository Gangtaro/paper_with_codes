from turtle import forward
import torch
import torch.nn as nn

# from .modules import Encoder, LayerNorm
from modules import Encoder, LayerNorm


class SASRecExplicit(nn.Module):
    def __init__(self, args):
        super(SASRecExplicit, self).__init__()
        self.item_embeddings = nn.Embedding(
            args.item_size, args.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction="none")
        self.apply(self.init_weights)

    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=sequence.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb
    
    def forward(self, input_items, input_ratings):

        # Attention Mask : mask after items
        attention_mask = (input_items > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64

        # Attention Weight Mask : weight Ratings to Attention score
        # input_ratings: sequence of ratings (1~5 -> 0~1)
        weighted_mask = input_ratings/5
        extended_weighted_mask = weighted_mask.unsqueeze(1).unsqueeze(2)

        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.tril(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        # Extend Attention mask to max length
        extended_attention_mask = extended_attention_mask * subsequent_mask        
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        # Extend weight mask to max length
        extended_weight_mask = extended_weighted_mask * subsequent_mask
        extended_weight_mask = extended_weight_mask.to(
            dtype=next(self.parameters()).dtype
        )

        # Sequential Embedding
        sequence_emb = self.add_position_embedding(input_items)


        item_encoded_layers = self.item_encoder(
            sequence_emb,                 # seqential emb
            extended_attention_mask,      # attention mask
            extended_weight_mask,         # weight mask (to attention score)
            output_all_encoded_layers=True
        )

        sequence_output = item_encoded_layers[-1]
        return sequence_output


    # def finetune(self, input_ids, input_ratings):
        
    #     # Explicit 으로 변경해주기 위해서는, attetion 할 때, 아이템에, 
    #     attention_mask = (input_ids > 0).long()
    #     weighted_mask = input_ratings/5 # Fixed - scaled
    #     extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
    #     extended_weighted_mask = weighted_mask.unsqueeze(1).unsqueeze(2)
    #     # extended_weighted_mask = torch.nn.functional.softmax((extended_weighted_mask == 0)*-10000.0 + extended_weighted_mask, dim=3)

    #     max_len = attention_mask.size(-1)
    #     attn_shape = (1, max_len, max_len)
    #     # subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
    #     subsequent_mask = torch.tril(torch.ones(attn_shape), diagonal=1)  # torch.uint8
    #     subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
    #     subsequent_mask = subsequent_mask.long()

    #     if self.args.cuda_condition:
    #         subsequent_mask = subsequent_mask.cuda()

    #     extended_attention_mask = extended_attention_mask * subsequent_mask        
    #     extended_attention_mask = extended_attention_mask.to(
    #         dtype=next(self.parameters()).dtype
    #     )  # fp16 compatibility
    #     extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    #     # Weight 
    #     extended_weight_mask = extended_weighted_mask * subsequent_mask
    #     extended_weight_mask = extended_weight_mask.to(
    #         dtype=next(self.parameters()).dtype
    #     )  # fp16 compatibility

    #     sequence_emb = self.add_position_embedding(input_ids)

    #     item_encoded_layers = self.item_encoder(
    #         sequence_emb, extended_attention_mask, extended_weight_mask, output_all_encoded_layers=True
    #     )

    #     sequence_output = item_encoded_layers[-1]
    #     return sequence_output

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
