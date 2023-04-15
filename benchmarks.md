# Benchmarks

Unscientific benchmark perplexities below for WikiText2. This follows the methodology described in the GPTQ and SparseGPT papers. and was generated via `ppl_benchmark.py`. See the old benchmarks section for more detail/disclaimers.

### LLaMA-7B

| Bits/W |   Config   |  ppl |
| :----: | :--------: | :--: |
|    16  |     fp16   |  OOM |
|  4.25  |  INT4 g128 | 5.82 |
|     4  |     INT4   | 6.10 |
|  3.25  |  INT3 g128 | 6.48 |
|     3  | INT4 16:32 | 7.85 |

Bits/W are effective bits per weight. All runs were done with staged calibration, activation sorting, and guessed sparsity percentage 0.5 (if applicable) on 128 calibration samples.

The arrangements represent the optimal configurations I've found so far (INT4 16:32 g128 in particular seems to underperform INT3 g128 for whatever reason.)


<details><summary>Old Benchmarks</summary>
<p>
Results are separated by base model I've tested. Note that your numbers may vary since quantization doesn't seem to be entirely deterministic (somehow.) Also, PPL seems to be very sensitive to the quality of calibration data as well. YMMV.

Note that the below benchmarks were current as-of 3c1c616, and predate some of the new quantization options.

### OPT

| Bits | Sparsity |  125M |  2.7B |
| :--: | :------: | :---: | :---: |
|  16  |   100%   | 27.66 | 12.46 |
|   4  |   100%   | 31.13 | 13.16 |
|   4  |  16:32   | 40.97 | 14.61 |

### LLaMA

| Bits | Sparsity |  7B  |  13B |
| :--: | :------: | :--: | :--: |
|  16  |   100%   |  OOM |  OOM |
|   4  |   100%   | 6.83 |  OOM |
|   4  |  16:32   | 9.53 | 6.78 |

### Pythia

| Bits | Sparsity |  350M | 1.3B  |
| :--: | :------: | :---: | :---: |
|  16  |   100%   | 16.66 | 12.68 |
|   4  |   100%   | 22.41 | 13.69 |
|   4  |  16:32   | 57.54 | 21.02 |

(not sure why sparsity hurts PPL so much here)
</p>
</details>