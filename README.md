# 학습 기록

## 1. Langchain을 사용할 때, 단계별 커스텀을 하는 방법?

- `Base` 클래스를 수정하면 된다.
- ex) 문서 Load를 디벨롭하고 싶다면 `BaseLoader`에서 서브클래싱하여 커스텀 구현을 하면 된다.
  - `from langchain_core.document_loaders import BaseLoader`
    ```python
    from typing import Iterator

    from langchain_core.document_loaders import BaseLoader
    from langchain_core.documents import  Document
    
    
    class CustomDocumentLoader(BaseLoader):
        def __init__(self, file_path: str) -> None:
            """
            내용 구현
            """
  
        def lazy_load(self) -> Iterator[Document]:
            """
            내용 구현
            """
    ```
- `yield`? 
  - 대용량 데이터를 처리할 때 한번에 많은 데이터를 불러와서 메모리에 올리는 것이 아니라 분할하여 메모리에 올린 후 처리하고 다시 내리는 식의 반복처리를 하게 하여 '지연 로딩'이 가능하게 하는 python 문법.
  - pytorch 등의 대용량 처리 라이브러리들도 모두 사용하는 개념임.