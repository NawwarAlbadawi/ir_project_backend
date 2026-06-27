import logging
import re
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from preprocessing_service.schemas import PreprocessOptions
logger = logging.getLogger('preprocessing_service.preprocessor')

def _ensure_nltk_data():
    packages = [('tokenizers/punkt_tab', 'punkt_tab'), ('corpora/stopwords', 'stopwords'), ('corpora/wordnet', 'wordnet'), ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'), ('corpora/omw-1.4', 'omw-1.4')]
    for path, pkg in packages:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info(f'Downloading NLTK package: {pkg}')
            nltk.download(pkg, quiet=True)
_WN_POS_MAP = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}

def _penn_to_wn(tag: str) -> str:
    return _WN_POS_MAP.get(tag[0].upper(), wordnet.NOUN)

class Preprocessor:

    def __init__(self):
        _ensure_nltk_data()
        self._stop_words: set[str] = set(stopwords.words('english'))
        self._stemmer = PorterStemmer()
        self._lemmatizer = WordNetLemmatizer()
        self._punct_table = str.maketrans('', '', string.punctuation)
        logger.info('Preprocessor initialised (stemmer + lemmatizer ready).')

    def preprocess(self, text: str, options: PreprocessOptions | None=None) -> list[str]:
        if options is None:
            options = PreprocessOptions()
        if options.lowercase:
            text = text.lower()
        text = re.sub('http\\S+|www\\.\\S+', ' ', text)
        text = re.sub('\\d+', ' ', text)
        if options.remove_punctuation:
            text = text.translate(self._punct_table)
        tokens: list[str] = word_tokenize(text)
        if options.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stop_words and len(t) > 1]
        if options.lemmatize:
            tokens = self._lemmatize_tokens(tokens)
        if options.stem:
            tokens = [self._stemmer.stem(t) for t in tokens]
        tokens = [t for t in tokens if t.strip()]
        return tokens

    def preprocess_to_string(self, text: str, options: PreprocessOptions | None=None) -> str:
        return ' '.join(self.preprocess(text, options))

    def _lemmatize_tokens(self, tokens: list[str]) -> list[str]:
        tagged = pos_tag(tokens)
        return [self._lemmatizer.lemmatize(word, pos=_penn_to_wn(tag)) for word, tag in tagged]