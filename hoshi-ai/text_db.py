import lancedb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from lancedb.rerankers import LinearCombinationReranker
from lancedb.pydantic import Vector, LanceModel
from lancedb.embeddings import EmbeddingFunctionRegistry
from dotenv import load_dotenv

load_dotenv()


class TextDB:
    def __init__(self, table_name: str):
        self.registry = EmbeddingFunctionRegistry().get_instance()
        self.cohere = self.registry.get("cohere").create(name="embed-english-v3.0")

        class Schema(LanceModel):
            text: str = self.cohere.SourceField()
            vector: Vector(self.cohere.ndims()) = self.cohere.VectorField()
            source: str

        self.db = lancedb.connect("./.lancedb")
        self.table = self.db.create_table(table_name, schema=Schema, exist_ok=True)
        self.table.create_fts_index("text", replace=True)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
        self.reranker = LinearCombinationReranker(weight=0.5)

    def add_text(self, text: str, source: str):
        text = self.splitter.create_documents([text])
        splits = self.splitter.split_documents(text)
        data = []
        for split in splits:
            chunk = split.page_content
            data.append({"text": chunk, "source": source})
        self.table.add(data)

    def search(self, text_query: str, k: int = 1):
        res = self.table.search(text_query, query_type="hybrid").limit(k).to_list()
        return res
