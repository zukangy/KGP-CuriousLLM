from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset


def parse_data(data):
    print("Parsing raw data...")
    titles = []
    passages = []
    for record in tqdm(data):
        title_chunks = record['title_chunks']
        for chunk in title_chunks:
            titles.append(chunk[0].strip())
            passages.append(chunk[1].strip())

    data_dict = {
        'title': titles,
        'passage': passages
    }
    
    data_df = pd.DataFrame(data_dict)
    
    # Remove duplicated passages
    data_df.drop_duplicates(subset=['passage'], ignore_index=True, inplace=True)
    data_df['passage_id'] = data_df.index
    
    data = data_df.to_dict(orient='records')
    print('Finished...')
    return data


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