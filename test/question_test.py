from vector_store import chroma_db
from model import embedding_model, remote_llm_model
from prompt import senario_prompt
from langchain.chains import RetrievalQA

pdf_name = 'dios.pdf'
collection_name = 'dios'
# question = '식기세척기 내부에 얼룩이 생겼는데 왜그런거야?'
question = '세척이 잘 안될때는 무슨 문제가 있는걸까?'

def print_split(text: str):
    print('-'*10 + text + '-'*10)

print_split('load_chroma_db')

embeddings_model = embedding_model.generate_huggingface_embedding_model()

find_db = chroma_db.find_db(
    embeddings_model,
    collection_name=collection_name,
)

print(f"find_db의 벡터 데이터의 collection_name: {find_db._collection.name}")
print(f"find_db의 벡터 갯수: {len(find_db)}")


print_split('generate prompt')

prompt_template = senario_prompt.generate_prompt_template()
# prompt = prompt_template.format(
#     question="우리나라와 미국, 중국, EU의 인공지능 산업 육성 정책의 특징을 정리해줘"
# )

print(f"-----------------로드한 프롬프트 탬플릿: \n{prompt_template}")
# print(f"-----------------생성한 프롬프트: \n{prompt}")

print_split('generate llm')

remote_llm = remote_llm_model.generate_remote_llm()

print_split('generate chain')

chain = prompt_template | remote_llm

print_split('retrieval chain start')

retriever = find_db.as_retriever(
    search_type="mmr",
    search_kwargs={
        'k': 3,
        'fetch_k': 10,
    }
)

qa = RetrievalQA.from_chain_type(
    llm=remote_llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

result = qa.invoke({"query": question})
print(result)

####################### 일반 llm에게 질문하여 답변받는것

# answer = chain.invoke({"question": question})
# print(answer)

# for token in chain.stream({"question": question}):
#     print(token['content'], end="")


# for token in chain.stream({"question": question}):
#     content = token.content
#     # Split the content by newline characters
#     lines = content.split('\n')
#     for i, line in enumerate(lines):
#         # Print each line
#         if i > 0:
#             print()  # Print a newline for each split
#         print(line, end="")
#     # To ensure any remaining text after the last newline character is printed correctly
#     if not content.endswith('\n'):
#         print(end="")

#################################