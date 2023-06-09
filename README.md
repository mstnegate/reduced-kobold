# reduced-kobold

This repo contains an implementation of GPTQ[^1] + SparseGPT[^2] with some [hacky] integration with KoboldAI and oobabooga's UI. Theoretically it should work with most stuff that uses HF transformers; I've just only tested with those.

Note that I am neither associated with these papers nor their authors.

## Supported Models/Configurations

Supported models:
* OPT models without project in/out layers; tested for 125M, 2.7B, 6.7B, 13B
* LLaMA (experimental); tested for 7B, 13B
* GPT-NeoX/Pythia; tested for Pythia 350M/1.3B

Support for other base models is planned.

The following configurations are supported for [accelerated] inferencing:
* GPTQ 4-bit dense (4-bit)
* GPTQ 4-bit with 16:32 sparsity (3-bit)
* GPTQ 3-bit dense (3-bit)

Group quantization is supported for all configurations.

## Benchmarks

Perplexity benchmarks can be [found under benchmarks.md](benchmarks.md).

## Setup

You will need to perform these steps under whichever Python env you run your stuff with.

1. Install CUDA quantized int4-fp16 matrix multiplication kernels: https://github.com/mstnegate/int4matmul_kernels

2. Clone this repo and put it somewhere you'll remember.

3. Prepare a model; you have two options:
    1. Quantize and/or sparsify your model yourself:
        - Currently all configuration is done in global vars at the top of `quantization.py`
        - Sharded model loading is supported in case you can't fit the model in memory [but have a sharded version of the model somehow.]
        - This only outputs a set of quantized tensors; you have to manually set up the model for usage with HF.
            - Copy everything in the source directory of your model (except for the weights) and create a new folder for your quantized model.
            - Copy over the quantized weights (`pytorch_model.bin` by default) into that new folder.
            - Modify config.json: change `model_type` from `"opt"` to `"opt-sq-reduced"`, add the key-value pair `"quantization_bits" : 4`, and optionally, `"is_sparse" : true` if you sparsified. If you set up group quantization, add a key-value pair `"quantization_group_size"` for whatever size you picked.
    2. Use a pre-quantized model:
        - LLaMA models pre-quantized with [qwopqwop200's excellent GPTQ repo](https://github.com/qwopqwop200/GPTQ-for-LLaMa) are incompatible without slight manual work due to some bad naming decisions on my part. These include the decapoda-research ones. There is a compatibility layer, but you need to some setup:
            - YMMV with this method; I didn't prepare those weights.
            - Note that this code path isn't actively tested and does not support all features of that repo.
            - Create a folder with appropriate config files (config.json, generation_config.json, etc., etc.)
            - Copy the quantized weights file (probably `something.pt`) into this folder and rename it to `pytorch_model.bin`.
            - Modify config.json following the steps above. Do not set `"is_sparse"`. Add the additional key-value pair: `"qwopqwop_mode" : true`

4. (Optional): run benchmarks to make sure you didn't mess up.
    - Currently only validation on Wikitext2 is supported, though testing on any arbitrary dataset/string data should be an easy change if you have a more specific dataset you want to evaluate.
    - Benchmarking is done via `ppl_benchmark.py`; you can configure what model to run in the file.

5. Set up integration for your front-end of choice. Note that these methods are very hacky right now.
    - KoboldAI (I've only tested this on an old version of Official.)
        - Somewhere near the top of `aiserver.py`, add `import hf_integration; hf_integration.register()`
        - You should be able to load your models through the normal UI; YMMV with LLaMA on Kobold, though; I couldn't get it running at all locally. OPT worked fine though.
        - You may need to disable lazy load
        - You may need to force this repo onto your path depending on how/where you set it up.
    - oobabooga's text-generation-ui:
        - Somewhere near the top of `modules/models.py`, add `import hf_integration; hf_integration.register()`.
        - You may need to force this repo onto your path depending on how/where you set it up.
        - You can load these models like standard models (rather than going through the --load-in-4-bit code path.)


[^1]: GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers; (https://arxiv.org/abs/2210.17323)
[^2]: SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot; (https://arxiv.org/abs/2301.00774)
