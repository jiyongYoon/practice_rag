from vector_store import chroma_db
from model import embedding_model, remote_llm_model, ollama_llm_model, openai_embeddings_model, openai_chatgpt_llm
from prompt import senario_prompt
from langchain.chains import RetrievalQA

def print_split(text: str):
    print('-'*10 + text + '-'*10)


print_split('setting variables')

pdf_name = 'dios.pdf'
# collection_name = 'dios'
# collection_name = 'dios-openai'
collection_name = 'safely_4-1'

# embedding_model_name = 'jhgan/ko-sbert-nli'
# embedding_model_name = 'jhgan/ko-sroberta-multitask'
# embedding_model_name = 'sentence-transformers/stsb-xlm-r-multilingual'

# question = '식기세척기 내부에 얼룩이 생겼는데 왜그런거야?'
# question = '린스 주입 방법을 알려줘'
question = '이 문서는 어떤 회사의 문서야?'

print_split('generate embeddings_model')

# embeddings_model = embedding_model.generate_huggingface_embedding_model(embedding_model_name, cuda_on=True)
embeddings_model = openai_embeddings_model.generate_embedding_model()

print_split('generate llm')

# remote_llm = remote_llm_model.generate_remote_llm()
llm = ollama_llm_model.llm
# llm = openai_chatgpt_llm.generate_llm()
# llm = ollama_llm_model.llama

print_split('load_chroma_db')


find_db = chroma_db.find_db(
    embeddings_model,
    collection_name=collection_name,
)

print(f"find_db의 벡터 데이터의 collection_name: {find_db._collection.name}")
print(f"find_db의 벡터 갯수: {len(find_db)}")

all_documents = chroma_db.get_all_documents(find_db, collection_name=collection_name)

# print(all_documents)

print_split('generate prompt')


prompt_template = senario_prompt.generate_safety_goal_text()
# prompt = prompt_template.format(
#     question="우리나라와 미국, 중국, EU의 인공지능 산업 육성 정책의 특징을 정리해줘"
# )

print(f"-----------------로드한 프롬프트 탬플릿: \n{prompt_template}")
# print(f"-----------------생성한 프롬프트: \n{prompt}")

print_split('retrieval chain start')

# ## 조금 다양한 답변
# retriever = find_db.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         'k': 2,
#         'fetch_k': 2,
#     }
# )

##
retriever = find_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        'score_threshold': 0.5,
        'k': 2,
    }
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

# result = qa.invoke({"query": question})
result = qa.invoke({"query": prompt_template})
print(result)

import json

converted_data = {
    'query': result['query'],
    'result': "",
    'source_documents': [
        {
            'page_content': doc.page_content,
            'metadata': doc.metadata
        } for doc in result['source_documents']
    ]
}

print(json.dumps(converted_data, ensure_ascii=False, indent=4))

print(f"""결과: \n{result['result']}""")

# source_documents_ = result['source_documents']
# for document in source_documents_:
#     print(document, end="\n")

# for token in qa.stream({"query": question}):
#     content = token['result']
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