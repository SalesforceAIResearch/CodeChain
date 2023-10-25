import os
from typing import Dict
import torch
from transformers import AutoModel, AutoTokenizer
from embedding.constants import SEPARATOR_TOKEN, CLS_TOKEN
from embedding.utils import pool_and_normalize
from embedding.datasets_loader import prepare_tokenizer
from embedding.preprocessing_utils import truncate_sentences
import pdb 
from tqdm import tqdm 

def set_device(inputs: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    output_data = {}
    for k, v in inputs.items():
        output_data[k] = v.to(device)
    
    return output_data

class BaseEncoder(torch.nn.Module): #, ABC):

    def __init__(self, device, max_input_len, maximum_token_len, model_name):
        super().__init__()

        self.model_name = model_name
        self.tokenizer = prepare_tokenizer(model_name)
        if model_name == 'Salesforce/codet5p-110m-embedding':
            self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval() 
        else:
            self.encoder = AutoModel.from_pretrained(model_name, use_auth_token=True).to(device).eval()
        self.device = device
        self.max_input_len = max_input_len
        self.maximum_token_len = maximum_token_len
    
    #@abstractmethod
    #def forward(self,):
    #    pass
    
    def encode(self, input_sentences, batch_size=32, **kwargs):
        truncated_input_sentences = truncate_sentences(input_sentences, self.max_input_len)

        n_batches = len(truncated_input_sentences) // batch_size + int(len(truncated_input_sentences) % batch_size > 0)

        embedding_batch_list = []

        for i in tqdm(range(n_batches)):
            start_idx = i*batch_size
            end_idx = min((i+1)*batch_size, len(truncated_input_sentences))

            with torch.no_grad():
                embedding_batch_list.append(
                    self.forward(truncated_input_sentences[start_idx:end_idx]).detach().cpu()
                )

        input_sentences_embedding = torch.cat(embedding_batch_list)
        return [emb.squeeze() for emb in input_sentences_embedding]

        #return [emb.squeeze().numpy() for emb in input_sentences_embedding]
    
class StarEncoder(BaseEncoder):

    def __init__(self, device, max_input_len, maximum_token_len):
        super().__init__(device, max_input_len, maximum_token_len, model_name = "bigcode/starencoder")
    
    def forward(self, input_sentences):

        inputs = self.tokenizer(
            [f"{CLS_TOKEN}{sentence}{SEPARATOR_TOKEN}" for sentence in input_sentences], 
            padding="longest",
            max_length=self.maximum_token_len,
            truncation=True,
            return_tensors="pt",
            )

        outputs = self.encoder(**set_device(inputs, self.device))
        embedding = pool_and_normalize(outputs.hidden_states[-1], inputs.attention_mask)
        
        return embedding
    
class CodeT5Plus(BaseEncoder):
    def __init__(self, device, max_input_len, maximum_token_len):
        super().__init__(device, max_input_len, maximum_token_len, model_name = "Salesforce/codet5p-110m-embedding")

        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-110m-embedding", trust_remote_code=True)

    def forward(self, input_sentences):

        inputs = self.tokenizer(
            [f"{CLS_TOKEN}{sentence}{SEPARATOR_TOKEN}" for sentence in input_sentences], 
            padding="longest",
            max_length=self.maximum_token_len,
            truncation=True,
            return_tensors="pt",
            )

        embedding = self.encoder(**set_device(inputs, self.device))

        return embedding
    
        
class CodeBERT(BaseEncoder):

    def __init__(self, device, max_input_len, maximum_token_len):
        super().__init__(device, max_input_len, maximum_token_len, model_name = "microsoft/codebert-base")

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    def forward(self, input_sentences):

        inputs = self.tokenizer(
            [sentence for sentence in input_sentences], 
            padding="longest",
            max_length=self.maximum_token_len,
            truncation=True,
            return_tensors="pt",
            )

        inputs = set_device(inputs, self.device)
        
        #print(inputs["input_ids"].shape, inputs["attention_mask"].shape)
        outputs = self.encoder(inputs["input_ids"], inputs["attention_mask"])

        embedding = outputs["pooler_output"]

        return torch.cat([torch.nn.functional.normalize(torch.Tensor(el)[None, :]) for el in embedding])

