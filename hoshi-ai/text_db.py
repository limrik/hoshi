import lancedb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from lancedb.rerankers import LinearCombinationReranker
from lancedb.pydantic import Vector, LanceModel
from lancedb.embeddings import EmbeddingFunctionRegistry
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()


def cosine_sim(str1, str2):
    vectorizer = CountVectorizer().fit_transform([str1, str2])
    vectors = vectorizer.toarray()
    cos_sim = cosine_similarity(vectors)
    return cos_sim[0][1]


class TextDB:
    def __init__(self, table_name: str):
        self.registry = EmbeddingFunctionRegistry().get_instance()
        self.openai = self.registry.get("openai").create(name="text-embedding-3-small")

        class Schema(LanceModel):
            text: str = self.openai.SourceField()
            vector: Vector(self.openai.ndims()) = self.openai.VectorField()
            source: str
            token_id: str

        self.db = lancedb.connect("./.lancedb")
        self.table = self.db.create_table(table_name, schema=Schema, exist_ok=True)
        self.table.create_fts_index("text", replace=True)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
        # self.reranker = LinearCombinationReranker(weight=0.2)  # Full text search is 0.8
        self.reranker = LinearCombinationReranker(weight=0.5)

    def add_text(self, text: str, source: str, token_id: str):
        text = self.splitter.create_documents([text])
        splits = self.splitter.split_documents(text)
        data = []
        for split in splits:
            chunk = split.page_content
            data.append({"text": chunk, "source": source, "token_id": token_id})
        self.table.add(data)

    def search(self, text_query: str, k: int = 1):
        res = self.table.search(text_query, query_type="hybrid").limit(k).to_list()
        out = []
        for item in res:
            source = item["source"]
            text = item["text"]
            score = cosine_sim(text_query, text)
            out.append({"source": source, "score": score, "token_id": item["token_id"], "content": text})
        return out
