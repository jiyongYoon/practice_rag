from model import remote_llm_model, ollama_llm_model
from langchain_core.prompts import ChatPromptTemplate


# llm = remote_llm_model.generate_remote_llm()
llm = ollama_llm_model.llm

prompt = ChatPromptTemplate.from_template(
    "다음 질문에 대해 한국말로 답변해주세요:\n{input}"
)

chain = prompt | llm

question = """
너는 정체가 뭐야?
"""

dic = {"input": question}

# answer = chain.invoke(dic)
# print(answer['content'])

for s in chain.stream(dic):
    print(s.content, end="", flush=True)
    # print(s['content'], end="", flush=True)
