from nltk.corpus import stopwords
import re
from src.data_transform import full_stopwords
from nltk import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer
from polyglot.tag import POSTagger as PG_POSTagger
from nltk import pos_tag as nltk_pos_tag
from lemmy import lemmatizer
from nltk.stem import WordNetLemmatizer as WNL
from nltk.corpus import wordnet
from string import punctuation
from sklearn.base import TransformerMixin
import pandas as pd
from polyglot.detect import Detector


class Preprocessor():

    def __init__(self, lang="en"):
        """
        """
        self.lang = lang
        self.lang_fullname = self._lang_fullname(lang)
        self.punctuation = list(punctuation)
        self.models = dict()

    def sentence_tokenize(self, text_list):
        """
        tokenize a list of documents into sentences [doc] -> [ [sentence] ].

        :param text_list: raw text corpus
        :rtype: Tokenized sentences
        """
        return [sent_tokenize(text, language=self.lang_fullname) for text in text_list]

    def word_tokenize(self, text_list):
        """
        tokenize a list of documents into words [doc] -> [ [word] ].

        :param text_list: raw text corpus
        :rtype: Tokenized words
        """
        return [word_tokenize(text, language=self.lang_fullname) for text in text_list]

    def stem(self, word_list):
        """
        stem words into root forms [word_conjugate] -> [word_root].

        :param text_list: raw text corpus
        :rtype: stemmed words
        """
        sno = SnowballStemmer(self.lang_fullname)
        return [sno.stem(word) for word in word_list]

    def pos_tag(self, word_list):
        """
        tag word by part-of-speech. [word] -> [ (word, tag) ].
        Choose between english (nltk) tagging or danish (polyglot) tagging.

        :param text_list: tokenized text corpus
        :rtype: POS tagged words
        """
        if (self.lang.lower() == "da"):
            tagger = PG_POSTagger(lang="da")
            return list(tagger.annotate(word_list))
        elif (self.lang.lower() == "se"):
            tagger = PG_POSTagger(lang="sv")
            return list(tagger.annotate(word_list))
        elif (self.lang.lower() == "en"):
            return nltk_pos_tag(word_list)
        else:
            pass

    def lemmatize(self, tag_list):
        """
        lemmatize word into canonical forms (NB. different from stemming)
        using tag information [ (word_conjugate, tag) ] -> [word_canonical].
        Choose between danish (lemmy) or english (nltk) lemmantizer.

        :param tag_list: tokenized word list
        :rtype: Lemmantized words
        """
        if (self.lang.lower() == "da"):
            lem = lemmatizer.load("da")
            return [lem.lemmatize(tag, word)[0] for word, tag in tag_list]
        elif (self.lang.lower() == "se"):
            lem = lemmatizer.load("sv")
            return [lem.lemmatize(tag, word)[0] for word, tag in tag_list]
        elif (self.lang.lower() == "en"):
            wnl = WNL()
            return [wnl.lemmatize(word, self._get_wordnet_pos(tag)) for word, tag in tag_list]

    def tag_and_lemmatize(self, word_list):
        """
        tag and lemmatize [ word_conjugate ] -> [word_canonical].

        :param text_list: tokenized words list
        :rtype: Lemmantized and tagged words using nltk
        """
        return self.lemmatize(self.pos_tag(word_list))

    def get_words_by_tag(self, tag_list, tags):
        """
        get words according to a tag list of desired tags.

        :param tag_list: A list of desired tags.
        example ["VB","CD","NN"] which keeps verbs, digits and nouns.
        :param tags: tagged words
        :rtype: tagged words in tag_list.
        """
        return [(word, tag) for (word, tag) in tag_list if tag in tags]

    def remove_stopwords(self, word_list, full=False):
        """
        remove common stopwords. user may specify whether
        to use the full word list or not [ word ] -> [ word ].

        :param word_list: a list of words contained in document
        :param full: if True use the full danish stoplist
        provided by the package, else nltk stopwords
        :rtype: Words not in stoplist
        """
        if full:
            sw = fsw.FULL_STOPWORDS[self.lang]
        else:
            sw = stopwords.words(self.lang_fullname)
        return [word for word in word_list if word not in sw]

    def remove_punctuation(self, word_list, punctuation=None):
        """
        remove punctuations. user may specify punctuation set.
        default set is string.punctuation: !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~. [ word ] -> [ word ]

        :param word_list: a list of words contained in document
        :rtype: word list without punctuation
        """
        if punctuation is None:
            punctuation = self.punctuation
        return [word for word in word_list if word not in punctuation]

    def remove_numbers(self, word_list):
        """
        remove any numbers in each word [word] -> [word].

        :param word_list: a list of words contained in document
        :rtype: word list without numbers
        """
        return [re.sub(r'\d+', '', word) for word in word_list]

    def remove_symbols(self, word_list, symbols):
        """
        remove user-specified symbols [word] -> [word].

        :param word_list: a list of words contained in document
        :param symbols: a list of symbols you need removed
        :rtype: word list without symbols you specified in list
        """
        return [word for word in word_list if word not in symbols]

    def remove_empty_words(self, word_list):
        """
        remove empty words, i.e. '' [word] -> [word].

        :param word_list: a list of words contained in document
        :rtype: a list of non-empty words
        """
        return [word for word in word_list if not word == ""]

    def regex_sub(self, pattern_replace, doc_list):
        """
        apply sequential regex substitutes for each document.

        :param pattern_replace: (list of) tuple of str: (pattern, replace)
        :param doc_list: list of document strings, [ doc ].
        :rtype:
        """
        return [self._regex_sub_doc(pattern_replace, doc) for doc in doc_list]

    def _regex_sub_doc(self, pattern_replace, doc):
        if isinstance(pattern_replace, tuple):
            pattern_replace = [pattern_replace]
        for p, r in pattern_replace:
            doc = re.sub(p, r, doc)
        return doc

    def _lang_fullname(self, short_name):
        """
        get full language name

        :param short_name: short name of language
        :rtype: full language name
        """
        if (short_name.lower() == "da"):
            return "danish"
        elif (short_name.lower() == "en"):
            return "english"
        else:
            return "english"

    def _get_wordnet_pos(self, tag):
        """
        get POStag in wordnet format
        """
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN


"""
The following classes make all the preprocessing methods pipelinable in sklearn

The methods taking list of words, [word], will now take list of list of words:
[ [ word ] ], so that they fit into a common pipeline procedure.
"""


class UnitTransformer():
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def fit(self, X, Y=None):
        return self

    def transform(self, X):
        return X


class Stemmer(UnitTransformer):
    def transform(self, word_list,):
        """
        input: list of list of words
        """
        return [self.preprocessor.stem(wl) for wl in word_list]


class WordTokenizer(UnitTransformer):
    def transform(self, text_list):
        return self.preprocessor.word_tokenize(text_list)


class SentenceTokenizer(UnitTransformer):
    def transform(self, text_list):
        return self.preprocessor.sentence_tokenize(text_list)


class POSTagger(UnitTransformer):
    def transform(self, word_list):
        """
        input: list of list of words
        """
        return [self.preprocessor.pos_tag(wl) for wl in word_list]


class Lemmatizer(UnitTransformer):
    def transform(self, tag_list):
        """
        input: list of list of (word, tag) tuples
        """
        return [self.preprocessor.lemmatize(tl) for tl in tag_list]


class WordByTag(UnitTransformer):
    def __init__(self, prep, tags):
        self.preprocessor = prep
        self.tags = tags

    def transform(self, tag_list):
        return [self.preprocessor.get_words_by_tag(tl, self.tags) for tl in tag_list]


class PunctuationRemover(UnitTransformer):

    def __init__(self, preprocessor, punctuation=None):
        UnitTransformer.__init__(self, preprocessor)
        self.punctuation = punctuation

    def transform(self, word_list):
        """
        input: list of list of words
        """
        return [self.preprocessor.remove_punctuation(wl, self.punctuation) for wl in word_list]


class StopwordRemover(UnitTransformer):
    def __init__(self, preprocessor, full=False):
        UnitTransformer.__init__(self, preprocessor)
        self.full = full

    def transform(self, word_list):
        """
        input: list of list of words
        """
        return [self.preprocessor.remove_stopwords(wl, self.full) for wl in word_list]


class NumberRemover(UnitTransformer):
    def transform(self, word_list):
        """
        input: list of list of words
        """
        return [self.preprocessor.remove_numbers(wl) for wl in word_list]


class SymbolRemover(UnitTransformer):
    def __init__(self, preprocessor, symbols):
        UnitTransformer.__init__(self, preprocessor)
        self.symbols = symbols

    def transform(self, word_list):
        """
        input: list of list of words
        """
        return [self.preprocessor.remove_symbols(wl, self.symbols) for wl in word_list]


class EmptywordRemover(UnitTransformer):
    def transform(self, word_list):
        """
        input: list of list of words
        """
        return [self.preprocessor.remove_empty_words(wl) for wl in word_list]


class RegexSub(UnitTransformer):
    def __init__(self, preprocessor, pattern_replace):
        self.preprocessor = preprocessor
        self.pattern_replace = pattern_replace

    def transform(self, doc_list):
        return self.preprocessor.regex_sub(self.pattern_replace, doc_list)


class FillNA(UnitTransformer):
    def __init__(self, preprocessor, filler="NaN"):
        self.preprocessor = preprocessor
        self.filler = filler

    def transform(self, doc_list):
        return pd.Series(doc_list).fillna(self.filler)


class RemoveNA(UnitTransformer):
    def transform(self, doc_list):
        return pd.Series(doc_list).dropna()


class FirstColumn(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, dataframe):
        if isinstance(dataframe, pd.DataFrame):
            return dataframe.iloc[:, 0]
        else:
            return dataframe[:, 0]


class SelectColumn(TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, dataframe):
        return dataframe[self.column]


class FillWithColumn(TransformerMixin):
    """
    fill column1 with column2
    """

    def __init__(self, column1, column2):
        self.column1 = column1
        self.column2 = column2

    def fit(self, X, y=None):
        return self

    def transform(self, dataframe):
        dataframe[self.column1] = dataframe[self.column1].fillna(dataframe[self.column2])
        return dataframe


class DocToSentence(UnitTransformer):
    """
    transform document list to senttence list, useful for e.g. word2vec training
    with sentence boundaries.
    [ doc ] -> [ sent ]
    can be switched on/off during fit and transform
    """

    def __init__(self, preprocessor, run_during_fit=True, run_during_transform=True):
        self.preprocessor = preprocessor
        self.run_during_fit = run_during_fit
        self.run_during_transform = run_during_transform

    def fit_transform(self, doc_list, y=None):
        if self.run_during_fit:
            return self._transform(doc_list)
        else:
            return doc_list

    def transform(self, doc_list):
        if self.run_during_transform:
            return self._transform(doc_list)
        else:
            return doc_list

    def _transform(self, doc_list):
        return [sent for sent_list in self.preprocessor.sentence_tokenize(
            doc_list) for sent in sent_list]


def detect_language(doc, text):
    """This simple function detects present lanuages
    in text dataframe

    :param doc: Á pandas dataframe with text
    :param text: Which column of the doc dataframe is
    the text
    :returns: A new column to the doc df called "languages"

    """
    language = []
    for document in doc[text]:
        det = Detector(document)
        language.append(det.language.code)
    doc["languages"] = language
    return doc


def _de_list(doc):
    print(doc)
    if not isinstance(doc, str):
        return doc[0]
    else:
        return doc
