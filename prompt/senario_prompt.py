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


def generate_safely_docs_prompt_template():
    template = """
너가 제공받은 문서는 중대재해처벌법 해설서 문서야. 질문에 대한 모든 답변은 이 안에서만 이루어져야 해.
창의적인 답은 절대로 하지 말고 자료에 있는 내용에서만 답변하고 없는 내용은 절대로 지어내지 않고 모른다고 답해.
한글로 답변해줘.

질문 내용은 아래와 같아.

{question}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["question"],
    )

    return prompt


def generate_safety_goal_text():
    template = """
너는 전달받은 문서에서 아래와 같은 항목의 내용이 있는지 찾아주는 역할을 해야해.
문서 내에서만 정보를 찾으며, 모르겠다면 지어내지 않고 문서 내에서는 찾을 수 없다고 답변해야해.
문서가 아래 항목을 만족하는지 확인하고, 항목을 만족하면 어느 부분이 그에 해당하는지, 그리고 얼마나 만족하는지를 알려줘.

[항목]
1. 회사의 안전보건에 관한 경영방침을 설정하고 있습니까?
2. 회사의 안전보건에 관한 목표를 설정하고 있습니까?

그리고 만약 질문에 대한 적합한 답변이 있다면 100점 만점에 총 몇 점인지 수치화해서 마지막에 말해줘.
항목이 2가지니까 각 항목당 완벽하게 만족한다면 50점이 되어서 합산하면 최대 100점이 될거야.
`총점: n 점 / 100 점` 이런 형태로 말이야
    """

    # prompt = PromptTemplate(
    #     template=template,
    #     input_variables=["question"],
    # )
    #
    # return prompt

    return template
