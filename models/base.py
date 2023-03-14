import json
import os

import torch
import transformers

class SQBaseModelWrapper:
    def __init__(self, fld):
        super().__init__()

        self.base_path = fld
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.base_path)

        # useful and small enough to just load for everyone
        with open(os.path.join(self.base_path, "config.json"), "rb") as f:
            self.model_conf = json.load(f)

class SQShardedMixin:
    def setup_sharding(self):
        # mildly inelegant way of avoiding deadly diamond of deaths in __init__
        with open(os.path.join(self.base_path, "pytorch_model.bin.index.json"), "rb") as f:
            self.map = json.load(f)

        self._loaded_containers = {}
        self.params_to_path = {}
        self.dont_save_these_keys = set()

        self.save_container = {}

    def _get_container(self, k):
        if k not in self._loaded_containers:
            self._loaded_containers[k] = torch.load(os.path.join(self.base_path, k))

        return self._loaded_containers[k]

    def get_weights_with_prefix(self, prefix):
        params_to_path = {}
        for k,v in self.map["weight_map"].items():
            if not k.startswith(prefix):
                continue

            params_to_path[k] = v

        return params_to_path

    def fetch_weights(self, k):
        if k not in self.map["weight_map"]:
            raise KeyError("Attempted to fetch unknown weight tensor %r" % k)

        k_file = self.map["weight_map"][k]

        container = self._get_container(k_file)
        return container[k]

    def flush_loaded(self):
        memo_keys = list(self._loaded_containers.keys())
        for k in memo_keys:
            del self._loaded_containers[k]

        # be very thorough
        del self._loaded_containers
        self._loaded_containers = {}
