from load_data import pdf_loader
from splitter import text_splitter
from model import embedding_model, openai_embeddings_model
from vector_store import chroma_db

### 테스트 환경별 변수값 세팅
pdf_name = 'bad2.pdf'
# collection_name = 'dios'
# collection_name = 'dios-openai'
collection_name = 'safely_4-1-bad2'

# embedding_model_name = 'jhgan/ko-sbert-nli'
# embedding_model_name = 'jhgan/ko-sroberta-multitask'
# embedding_model_name = 'sentence-transformers/stsb-xlm-r-multilingual'

# embeddings_model = embedding_model.generate_huggingface_embedding_model(embedding_model_name, cuda_on=True)
embeddings_model = openai_embeddings_model.generate_embedding_model()

def print_split(text: str):
    print('-'*10 + text + '-'*10)


print_split('pdf_load_start')

# pdf = pdf_loader.load_pdf('ai.pdf') ## Document 객체
pdf = pdf_loader.load_pdf(pdf_name) ## Document 객체
print(f"--------------------------전체 pdf 원문(document): \n{pdf}")

full_text = ''
str_list = []
for i in range(len(pdf)):
    str_list.append(pdf[i].page_content)
full_text = full_text.join(str_list)
print(f"--------------------------전체 pdf 원문(text): \n{full_text}")

print_split('text_split_start')

# splitter.split_documents를 하면 {page_contents, metadata{source, page}} 형태의 데이터가 나온다
# 이러한 split이 나중에 출처를 찾기에는 좋겠다
"""
{
  "page_content":"l 인력\n- 노동자 보호를 위한 정부 조정 역할 및 민간 참여 독려\n- 고급 개발 인력 확충 위한 STEM 교육정책 강화\nl 기타",
  "metadata":{
    "source":"D:\\wizcore\\project\\python\\safely_biz_test\\ai.pdf",
    "page":26
  }
}
"""
splitter = text_splitter.generate_text_splitter(500, 100)
split_documents = splitter.split_documents(pdf)
print_split('split_documents')
for num in range(len(split_documents)):
    print(split_documents[num], end='\n')
    print('-'*50)

split_text = splitter.split_text(full_text)
print_split('split_text')
for num in range(len(split_text)):
    print(split_text[num], end='\n')
    print('-'*50)

print_split('(Optional) embeddings start')
print_split('vector 저장소에 저장할때는 저장 시 embedding을 시켜서 저장하는 것이 한 process')


# embeddings = embeddings_model.embed_documents(split_documents)
# 위 메서드에서 Document 객체는 임베딩이 안됨. 'Document' object has no attribute 'replace'
# embeddings = embeddings_model.embed_documents(full_text)
#
# print(
#     len(embeddings),
#     len(embeddings[0]),
#     embeddings[0]
# )
# """
# 29595 768 [-0.007854025810956955, -0.08216159045696259...]
# """

print_split('store vectors start')

save_db = chroma_db.save_document(
    embedding_model=embeddings_model,
    document=split_documents,
    collection_name=collection_name,
)

print(f"저장된 벡터 데이터의 collection_name: {save_db._collection.name}")
print(f"저장된 벡터 데이터의 갯수: {len(save_db)}")
