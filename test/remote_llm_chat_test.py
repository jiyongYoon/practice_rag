from model import remote_llm_model
from langchain_core.prompts import ChatPromptTemplate


llm = remote_llm_model.generate_remote_llm()

prompt = ChatPromptTemplate.from_template(
    "다음 질문에 대해 한국말로 답변해주세요:\n{input}"
)

chain = prompt | llm

question = """
llm과 관련있는 LangChain의 LangSmith는 어떻게 사용하는거야?
"""

dic = {"input": question}

# answer = chain.invoke(dic)
# print(answer['content'])

for s in chain.stream(dic):
    print(s['content'], end="", flush=True)
