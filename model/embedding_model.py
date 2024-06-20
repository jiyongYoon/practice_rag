# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


def generate_huggingface_embedding_model(cuda_on: bool = None):
    device = 'cuda' if cuda_on else 'cpu'
    print(f"embedding 모델에 {device}를 사용합니다.")

    model_name = "jhgan/ko-sbert-nli"
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True}
    model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return model

