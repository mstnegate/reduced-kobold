import torch

import transformers
import datasets

import layers
import hf_integration
hf_integration.register()

################################################################################

MODEL_TO_EVALUATE = "/pth/to/your/model/folder"

################################################################################

# TODO: set up option to test on validation shard of C4 (following GPTQ/SparseGPT methodologies)

def load_data():
    raw_training_data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    collated_text = "\n\n".join(raw_training_data["text"])

    return collated_text

def load_model_and_tokenizer(spec):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        spec,
        torch_dtype=torch.float16,
    ).to("cuda:0")

    tokenizer = transformers.AutoTokenizer.from_pretrained(spec)

    return model, tokenizer

def ppl_calc(model, tokenizer, dataset):
    full_token = torch.tensor([tokenizer.encode(dataset)])

    sliding_window_size = 2048
    sliding_window_stride = 2048

    negloglikes = []
    for i in range(0, full_token.shape[-1], sliding_window_stride):
        print("%d/%d" % (i, full_token.shape[-1]))
        in_tokens = full_token[:, i:(i+sliding_window_size)].clone()
        in_tokens = in_tokens.to(model.device)

        with torch.no_grad():
            probs = model(in_tokens, labels=in_tokens)
            negloglikes.append(probs.loss.item() * in_tokens.shape[-1])

        del in_tokens
        del probs

    import numpy as np
    print("PPL: %.5f" % np.exp(sum(negloglikes) / full_token.shape[-1]))


if __name__ == "__main__":
    dataset = load_data()
    import time

    model, tokenizer = load_model_and_tokenizer(MODEL_TO_EVALUATE)
    t = time.time()
    ppl_calc(model, tokenizer, dataset)
    print("Time taken: %.5f" % (time.time() - t))
