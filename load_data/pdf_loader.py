import os
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv


def load_pdf(pdf_filename: str):
    load_dotenv()
    project_directory = os.environ['PROJECT_DIRECTORY']

    pdf_filepath = os.path.join(project_directory, pdf_filename)

    loader = PyPDFLoader(pdf_filepath)
    pages = loader.load_and_split()
    return pages
