

from transformers import AutoTokenizer, AutoModel
import torch

#bge-large-zh-v1.5
class embeddingmodel():

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()

    def batch_encode(self, texts: list[str]) -> list[list[float]]:

        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.numpy()


# if __name__=="__main__":
#     emb_model=embeddingmodel("/data0/models/embeddingmodels/bge-base-en")
#     a=emb_model.batch_encode("nihao!")
#     print(a.shape)
