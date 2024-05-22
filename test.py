from pinecone import Pinecone
from openai import OpenAI

client = OpenAI(
    api_key="OPENAI_API_KEY"
)  # get API key from platform.openai.com


MODEL = "text-embedding-3-small"

res = client.embeddings.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], model=MODEL
)

# we can extract embeddings to a list
embeds = [record.embedding for record in res.data]
len(embeds)

from datasets import load_dataset

# load the first 1K rows of the TREC dataset
trec = load_dataset('trec', split='train[:1000]')

PINECONE_API_KEY = "d203c250-37c6-4e38-8d0a-f2e4d4d4bd1b"

pc = Pinecone(api_key=PINECONE_API_KEY)

pc.list_indexes()