## Benchmarks

Unscientific benchmark perplexities are below for WikiText2. This follows the methodology described in the GPTQ and SparseGPT papers.

Results are separated by base model I've tested. Note that your numbers may vary since quantization doesn't seem to be entirely deterministic (somehow.) Also, PPL seems to be very sensitive to the quality of calibration data as well. YMMV.

### OPT

| Bits | Sparsity |  125M |  2.7B |
| :--: | :------: | :---: | :---: |
|  16  |   100%   | 27.66 | 12.46 |
|   4  |   100%   | 31.13 | 13.16 |
|   4  |  16:32   | 40.97 | 14.61 |

### LLaMA

| Bits | Sparsity |  7B  |
| :--: | :------: | :--: |
|  16  |   100%   |  OOM |
|   4  |   100%   | 6.83 |
|   4  |  16:32   | 9.53 |

### Pythia

| Bits | Sparsity |  350M | 1.3B  |
| :--: | :------: | :---: | :---: |
|  16  |   100%   | 16.66 | 12.68 |
|   4  |   100%   | 22.41 | 13.69 |
|   4  |  16:32   | 57.54 | 21.02 |

(not sure why sparsity hurts PPL so much here)