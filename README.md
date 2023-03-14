# reduced-kobold

This repo contains an implementation of GPTQ[^1] + SparseGPT[^2] with some [hacky] integration with KoboldAI. Theoretically it should work with most stuff that uses HF transformers, but I only tested with Kobold.

Note that I am neither associated with these papers nor their authors.

The code supports int4 quantization via GPTQ, with optional 16:32 joint sparsification. Group quantization is currently not supported.

Note that sparsification is currently experimental. Benchmark results loosely follow the paper's claims but actual generation can be tempermental. I haven't ran these enough to comment on output quality relative to lower param count dense models. Hopefully my code is just bugged!

## Supported Models/Configurations

This repo supports OPT-based models without project in/out layers; I've tested for 125M, 2.7B, 6.7B, and 13B.

It has also experimental support for LLaMA models. I've only tested for 7B currently. Note that you will need to install a non-standard version of transformers.

Support for other base models is planned.

This repo only contains [accelerated] support for GPTQ int4 and GPTQ int4 + 16:32 sparsified models. This should provide 4-bit and 3-effective-bit weights for [most] of the model.

## Benchmarks

Unscientific benchmark perplexities are below for WikiText2. This follows the methodology described in the GPTQ and SparseGPT papers.

| Bits | Sparsity | OPT-125M | OPT-2.7B | LLaMA-7B |
| :--: | :------: | :------: | :------: | :------: |
|  16  |   100%   |   27.66  |   12.46  |    OOM   |
|   4  |   100%   |   31.13  |   13.16  |   6.83   |
|   4  |  16:32   |   40.97  |   14.61  |   9.53   |

Note that your numbers may vary since quantization doesn't seem to be entirely deterministic (somehow.) Also, PPL seems to be very sensitive to the quality of calibration data as well. YMMV.

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
            - Modify config.json: change `model_type` from `"opt"` to `"opt-sq-reduced"`, add the key-value pair `"quantization_bits" : 4`, and optionally, `"is_sparse" : true` if you sparsified.
    2. Use a pre-quantized model:
        - LLaMA models pre-quantized with [qwopqwop200's excellent GPTQ repo](https://github.com/qwopqwop200/GPTQ-for-LLaMa) are incompatible without slight manual work due to some bad naming decisions on my part. These include the decapoda-research ones. There is a compatibility layer, but you need to some setup:
            - YMMV with this method; I didn't prepare those weights.
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
