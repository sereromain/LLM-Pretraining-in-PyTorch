from dataclasses import dataclass

@dataclass
class LargeLanguageModelConfig:
    seq_size: int = 1024    # size of the tokens sequence
    vocab_size: int = 50257 # GPT-2 vocab_size of 50257
    nb_trans: int = 12      # number of Transformer layers
    nb_head: int = 12       # number of heads for MultiHeadSelfAttention
    embed: int = 768        # size of the embedding
    dropout: float = 0.0    # amont of dropout
    bias: bool = True       # True: bias in Linears and LayerNorms, like GPT-2
    fast_att: bool = True   # True: the model will use fast-attentionv2 implemented in Pytorch

@dataclass
class TrainingConfig:
    num_epochs: int = 500 # number of epochs
    learning_rate: float = 5e-4 # initial learning rate
    epoch_train_steps: int = 50 # number of training batches called inside an epoch
    epoch_val_steps: int = 5 # number of validation batches called inside an epoch
    batch_size: int = 480 # sequences needed before gradient update & metrics evaluation
    accum_steps: int = 10  # accumulates gradients in this amount of steps (do this if the batch size is too large to fit in VRAM)
    lr_patience: int = 10 # ReduceLROnPlateau number of epochs before changing the learning rate
    lr_multiplier: float = 0.5 # ReduceLROnPlateau multiplier to change the learning rate
    data_dir: str = './data' # directory containing the train.bin and val.bin datasets