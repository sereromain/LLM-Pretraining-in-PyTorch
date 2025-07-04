import os
import torch
import tiktoken

from model import LargeLanguageModel


weights_file = "llm.pt"


print('Enter your prompt :')
prompt = input()

# prompt = "Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist who is best known for developing the theory of relativity. Einstein also made important contributions to quantum mechanics. His mass–energy equivalence formula E = mc2, which arises from special relativity, has been called the worlds most famous equation. He received the 1921 Nobel "

model = LargeLanguageModel().to('cuda')
# model.bfloat16()

if os.path.isfile(weights_file):
    model.load_state_dict(torch.load(weights_file, weights_only=True))
else:
    print("[ERROR] The model weights are not present in " + weights_file)

model.eval()
enc = tiktoken.get_encoding("gpt2")
rep = enc.encode(prompt)
encoded_prompt = torch.Tensor([rep]).long().to('cuda')
generated_prompt = model.generate(encoded_prompt,max_new_tokens=500)