# Copyright 2025-2026 Muhammad Nizwa. All rights reserved.

import torch
from easydict import EasyDict


def get_default_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_device(data, device=get_default_device()):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# model config
en_id_model = EasyDict(__name__="Config: En Id Model Translation")

# data config
en_id_model.corpus = "opus100" # Huggingface corpus name
en_id_model.lang_src = "en"
en_id_model.lang_tgt = "id"
en_id_model.tokenizer_file = ".output/tokenizer_{}.json" # trained tokenizer file path
en_id_model.train_set_ratio = 1 # train split factor if corpus is too large

# model config
#
# number of encoder & decoder layers (depends on seq_len)
# longer seq_len = more layers to capture long-range dependencies
en_id_model.num_layers = 4 
# number of attention heads (must be divisor of d_model: d_model // h)
# more heads = more capturing of different representation subspaces (required in translation)
# also consider d_model, if large heads with fewer d_model = each head has small dim
en_id_model.num_heads = 4 
# dimension of model, 512 is standard
# represent the dim of each token embedding, more dim = represent more meaning (required in translation)
en_id_model.d_model = 256 
# dimension of feedforward network, usually: 4 * d_model
en_id_model.ffn_dim = 1024 
# overall dropout rate (adjustable)
# can try different combination: 0.05, 0.1, 0.15
en_id_model.dropout = 0.1 
# maximum sequence length the model can handle (simple translation = short sequence)
# any pairs longer than this seq_len will be pruned out during preprocessing
en_id_model.seq_len = 50 

# training config
en_id_model.batch_size = 64 # adjust based on your GPU memory
en_id_model.num_epochs = 30 # number of training epochs
en_id_model.lr = 0.0001 # learning rate
en_id_model.model_output = ".output/opus100_en_id_mtmodel.pth" # model checkpoint file path
