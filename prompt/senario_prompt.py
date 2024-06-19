from langchain_core.prompts import PromptTemplate


def generate_prompt_template():
    template = """
너는 자료에서 내가 궁금해하는 질문에 대한 답을 찾아주는 사람이야.
창의적인 답은 하지 말고 자료에 있는 내용에서만 답변해줘.
한글로 답변해줘.
주어진 자료에 없는 내용은 지어내지 않고 모른다고 답하면 돼.
내가 궁금한 내용은 아래와 같아.

{question}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["question"],
    )

    return prompt
