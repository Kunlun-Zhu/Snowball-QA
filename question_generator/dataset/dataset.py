from .utils import preprocess, read_json

class Seq2SeqDataset:
    def __init__(self, data_dir):
        self.datas = read_json(data_dir)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        return self.datas[index]


class DataCollatorForSeq2Seq:
    def __init__(self, tokenizer, padding: bool = True, max_length: int = 512):
        self.tokenizer = tokenizer
        #self.model = model
        self.padding = padding
        self.max_length = max_length

    def __call__(self, batch):
        features = self.collator_fn(batch)
        return features

    def collator_fn(self, batch):
        results = map(preprocess, batch)
        inputs, targets, _ = zip(*results)

        input_tensor = self.tokenizer(inputs,
                                      truncation=True,
                                      padding=True,
                                      max_length=self.max_length,
                                      return_tensors="pt",
                                      )

        target_tensor = self.tokenizer(targets,
                                       truncation=True,
                                       padding=True,
                                       max_length=self.max_length,
                                       return_tensors="pt",
                                       )

        input_tensor["labels"] = target_tensor["input_ids"]

        if "token_type_ids" in input_tensor:
            del input_tensor["token_type_ids"]
        return input_tensor
