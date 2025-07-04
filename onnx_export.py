import os
import torch
import tiktoken

from model import LargeLanguageModel


weights_file = "llm.pt"

model = LargeLanguageModel().to('cuda')
model.bfloat16()

if os.path.isfile(weights_file):
    model.load_state_dict(torch.load(weights_file, weights_only=True))
else:
    print("[ERROR] The model weights are not present in " + weights_file)

# Save the model weights and topology to ONNX format
model.eval().to('cpu')
fake_inputs = (torch.randint(model.config.vocab_size, (1, model.config.seq_size)),)
onnx_program = torch.onnx.export(model, fake_inputs, dynamo=True)
onnx_program.optimize()
onnx_program.save(weights_file[:-2]+"onnx")