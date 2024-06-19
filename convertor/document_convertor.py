from langchain.schema import Document


def change_document(clean_text: str):
    return [Document(page_content=clean_text)]
