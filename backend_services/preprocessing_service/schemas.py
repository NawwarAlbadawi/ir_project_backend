from pydantic import BaseModel, Field
from typing import Optional

class PreprocessOptions(BaseModel):
    lowercase: bool = Field(True, description='Convert text to lowercase.')
    remove_punctuation: bool = Field(True, description='Strip punctuation characters.')
    remove_stopwords: bool = Field(True, description='Remove English stop-words.')
    stem: bool = Field(True, description='Apply Porter Stemmer.')
    lemmatize: bool = Field(False, description='Apply WordNet Lemmatizer. If both stem and lemmatize are True, lemmatization runs first.')

class TextRequest(BaseModel):
    text: str = Field(..., description='The text to preprocess.')
    options: Optional[PreprocessOptions] = Field(None, description='Preprocessing options. Defaults applied if not provided.')

class BatchTextRequest(BaseModel):
    texts: list[str] = Field(..., description='List of texts to preprocess.')
    options: Optional[PreprocessOptions] = Field(None)

class TextResponse(BaseModel):
    original: str
    tokens: list[str]
    cleaned: str

class BatchTextResponse(BaseModel):
    results: list[TextResponse]

class DatasetStatusResponse(BaseModel):
    dataset: str
    status: str
    progress_docs: int = 0
    total_docs: int = 0
    progress_queries: int = 0
    total_queries: int = 0
    error: Optional[str] = None