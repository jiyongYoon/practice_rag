from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank


def load_rerank_retriever(base_retriever):
    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

    return compression_retriever

# reference: https://python.langchain.com/v0.2/docs/integrations/retrievers/flashrank-reranker/