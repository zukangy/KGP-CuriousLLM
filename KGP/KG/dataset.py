import json 
from torch.utils.data import Dataset


def parse_data(data):
    print("Parsing raw data...")
    passages = []
    p_id = 0 
    for q_id, record in enumerate(data):
        title_chunks = record['title_chunks']
        
        for chunk in title_chunks:
            title = chunk[0].strip()
            passage = chunk[1].strip()
            passages.append({
                'question_id': q_id,
                'passage_id': p_id,
                'title': title,
                'passage': passage
                }) 
            p_id += 1
    print('Finished...')
    return passages


class DocumentsDataset(Dataset):
    def __init__(self, raw_docs):
        super().__init__()
        self.docs = raw_docs

    def __getitem__(self, index):
        doc = self.docs[index]
        return doc['passage_id'], doc['title'], doc['passage']

    def __len__(self):
        return len(self.docs)
    
    
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings

    def __getitem__(self, index):
        return index, self.embeddings[index]

    def __len__(self):
        return len(self.embeddings)