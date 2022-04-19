import numpy as np
import pandas as pd
import scipy as sp
import sklearn
from pyemd import emd

from collections import defaultdict
from enum import Enum

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

from gensim.models import KeyedVectors
import gensim.downloader as api

from scipy.sparse import dok_matrix
from tqdm import tqdm
from scipy.sparse.linalg import svds
import altair as alt

################################################################################
#                            UTILS FUNCTIONS                                   #
################################################################################

def cast_list_as_strings(mylist):
    """
    return a list of strings
    """
    #assert isinstance(mylist, list), f"the input mylist should be a list it is {type(mylist)}"
    mylist_of_strings = []
    for x in mylist:
        mylist_of_strings.append(str(x))

    return mylist_of_strings


def get_features_from_df(df, count_vectorizer):
    """
    returns a sparse matrix containing the features build by the count vectorizer.
    Each row should contain features from question1 and question2.
    """
    q1_casted =  cast_list_as_strings(list(df["question1"]))
    q2_casted =  cast_list_as_strings(list(df["question2"]))

    ############### Begin exercise ###################
    # what is kaggle                  q1
    # What is the kaggle platform     q2
    X_q1 = count_vectorizer.transform(q1_casted)
    X_q2 = count_vectorizer.transform(q2_casted)
    X_q1q2 = sp.sparse.hstack((X_q1,X_q2))
    ############### End exercise ###################

    return X_q1q2


def get_mistakes(clf, X_q1q2, y):

    ############### Begin exercise ###################
    predictions = clf.predict(X_q1q2)
    incorrect_predictions = predictions != y
    incorrect_indices,  = np.where(incorrect_predictions)

    ############### End exercise ###################

    if np.sum(incorrect_predictions)==0:
        print("no mistakes in this df")
    else:
        return incorrect_indices, predictions


def preprocess(strings):
    """
    Takes a list and preprocesses and tokenizes strings
    """
    transform = CountVectorizer().build_analyzer()
    return [transform(str(s)) for s in strings]


def bag_of_words(documents, vocabulary, normalize=False):
    """
    Function that computes the BOW representation of a collection of documents.
    """
    data = []
    ind_col = []
    ind_ptr = [0]

    for doc in documents:
        # Create a bag of words representation for the document
        bow_doc = defaultdict(int)

        for word in doc:
            if word in vocabulary:
                bow_doc[word] += 1

        # Normalize BoW
        bow_array = np.array(list(bow_doc.values()))
        bow_norm = np.sum(bow_array) if normalize else 1.0

        bow_doc_normalized = [
            bow_doc[word] / bow_norm
            for word in bow_doc.keys()
        ]

        # Get columns in which the data will be stored
        cols = [vocabulary[word] for word in bow_doc.keys()]

        data.extend(bow_doc_normalized)
        ind_col.extend(cols)
        ind_ptr.append(len(ind_col))

    return np.array(data), np.array(ind_col), np.array(ind_ptr)

def tpr_fpr(p, dist, y_true):
    """
    return the true positive rate and false positive rate for a distance
    at a certain threshold
    """
    accurate = y_true == (dist < p)
    tpr = accurate[y_true == 1].mean()
    fpr = 1 - accurate[y_true == 0].mean()
    return tpr, fpr

################################################################################
#                         CUSTOM VECTORIZERS                                   #
################################################################################

class TfIdfCustomVectorizer(BaseEstimator, TransformerMixin):
    def _fit(self, X):
        n_docs = len(X)

        i = 0
        self.vocabulary = {}
        word_counts = defaultdict(int)
        
        for doc in X:
            words_in_document = set()

            for word in doc:
                if word not in self.vocabulary:
                    self.vocabulary[word] = i
                    i += 1
                
                words_in_document.add(word)

            for word in words_in_document:
                word_counts[word] += 1
        
        word_count_array = np.zeros(len(self.vocabulary))
        
        for word, idx in self.vocabulary.items():
            word_count_array[idx] = word_counts[word]
        
        # NOTE: This is the smoothed formula for the inverse document frequency
        # This way, the lower bound of the idf is 0.
        # sklearn uses idf = log((1 + |X|) / (1 + |X_w|)) + 1
        self.idf = np.log((n_docs) / (1 + word_count_array)) + 1


    def fit(self, X):
        X_processed = preprocess(X)
        self._fit(X_processed)

        return self
    

    def transform(self, X):
        X_preprocessed = preprocess(X)
        
        data, ind_col, ind_ptr = bag_of_words(X_preprocessed, self.vocabulary)

        for i in range(len(ind_ptr) - 1):
            # Get indices of current row and columns in current row
            current_idx, next_idx = ind_ptr[i], ind_ptr[i+1]
            cols = ind_col[current_idx:next_idx]

            bow_doc = data[current_idx:next_idx]
            bow_doc_tfidf = bow_doc * self.idf[cols]

            doc_norm = np.sqrt(np.dot(bow_doc_tfidf, bow_doc_tfidf))
            bow_doc_norm = bow_doc_tfidf / doc_norm

            data[current_idx:next_idx] = bow_doc_norm

        X_transformed = sp.sparse.csr_matrix(
            (data, ind_col, ind_ptr),
            shape=(len(X), len(self.vocabulary))
        )

        return X_transformed
    

    def fit_transform(self, X):
        self.fit(X)
        X_transformed = self.transform(X)

        return X_transformed


class EmbeddingType(str, Enum):
    WORD2VEC = 'embeddings/word2vec-google-news-300.kv'
    COOCCURRANCE_SVD = 'embeddings/kv_cooccurrance_svd.kv'


class TfIdfEmbeddingVectorizer(TfIdfCustomVectorizer):
    def __init__(self, embeddings_type=None):
        super().__init__()

        self.embeddings_type = embeddings_type

        # Load embeddings
        if self.embeddings_type is None:
            self.model = api.load('word2vec-google-news-300')
        else:
            self.model = KeyedVectors.load(self.embeddings_type)


    def fit(self, X):
        X_preprocessed = preprocess(X)

        # Generate vocabulary and idf values
        self._fit(X_preprocessed)

        # Generator used for indexing
        def index_generator(max_idx):
            idx = 0

            while idx < max_idx:
                yield idx
                idx += 1

        reindexer = index_generator(len(self.vocabulary))

        # Postprocess idf values by keeping the valid ones
        idf_valid_idx = [
            self.vocabulary[word]
            for word in self.vocabulary.keys()
            if word in self.model.key_to_index
        ]

        self.idf = self.idf[idf_valid_idx]

        # Postprocess vocabulary by keeping only the words that appear in
        # the model
        # The generator is used as kind of a counter variable, which saves
        # the need of a Python for loop
        self.vocabulary = {
            word: next(reindexer)
            for word in self.vocabulary.keys()
            if word in self.model.key_to_index
        }

        return self


    def transform(self, X):
        X_preprocessed = preprocess(X)
        X_transformed = []
        
        for doc in X_preprocessed:
            # Create a bag of words representation for the document
            # It will contain the sum of the tf-idf values
            bow_doc = defaultdict(float)

            for word in doc:
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    bow_doc[word] += self.idf[idx]

            # Normalize BoW
            bow_array = np.array(list(bow_doc.values()))
            bow_norm = np.sqrt(np.dot(bow_array, bow_array))

            bow_doc_normalized = {
                word: bow_doc[word] / bow_norm
                for word in bow_doc.keys()
            }

            # Compute sentence embedding as weighted sum of word embeddings
            doc_embedding = np.zeros(self.model.vector_size)

            for word, tfidf in bow_doc_normalized.items():
                doc_embedding = doc_embedding + self.model[word] * tfidf

            X_transformed.append(doc_embedding)
        
        X_transformed = np.array(X_transformed)

        return X_transformed

    

    
class cooccurrance_embeddings():
    def __init__(self, corpus, preprocess_function):
        self.corpus = corpus
        self.preprocess = preprocess_function
    
    def fit(self, n_components=100, store=True):
        print("computing coocurrance matrix")
        self.compute_cooccurrance()
        print("SVM decomposition")
        self.svm(n_components, store)

        return self

    def compute_cooccurrance(self):
        
        # Fit count vectorizer
        count_vectorizer = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(1,1))
        count_vectorizer.fit(self.corpus)
        self.idx = count_vectorizer.vocabulary_
              
        # Preprocess questions
        self.corpus = self.preprocess(self.corpus)
        
        # Initialize sparse squared empty matrix to store
        # co-occurrance between words in our vocabulary
        self.cOc_embd = dok_matrix((len(self.idx), len(self.idx)), dtype=np.int8)
        
        # Compute co-occurrances
        for question in tqdm(self.corpus):
            word_ids = list(set([self.idx[word] for word in question]))
            self.cOc_embd[word_ids,word_ids] += 1
        
        self.cOc_embd = self.cOc_embd.asfptype()
        
            
    def svm(self, n_components, store):
        self.svm_results = {}
        self.svm_results['u'], self.svm_results['s'], self.svm_results['vT'] = svds(self.cOc_embd, k=n_components)
        
        # Join with words indices
        idx_df = pd.DataFrame({'words': self.idx.keys(), 'key': self.idx.values()}, index=self.idx.values())
        self.svm_loadings = pd.DataFrame(self.svm_results['u'], index=[i+1 for i in range(0,len(self.idx))])
        self.svm_loadings = idx_df.merge(self.svm_loadings, left_index=True, right_index=True)

        # Store as keyed vectors
        self.svm_model = KeyedVectors(self.svm_results['u'].shape[1])
        self.svm_model.add_vectors(self.svm_loadings.words.values, self.svm_loadings.iloc[:,2:self.svm_loadings.shape[1]].to_numpy())
        
        if store:
            self.svm_model.save('embeddings/kv_coocurrance_svm.kv')
            
            
    def pca_2q_plot(self, q1, q2):
        q1, q2 = self.preprocess(q1), self.preprocess(q2)
        pca_df = pd.DataFrame({'Word': q1+q2,
                        'PC1': np.r_[self.svm_model[q1][:,0],self.svm_model[q2][:,0]],
                        'PC2': np.r_[self.svm_model[q1][:,0],self.svm_model[q2][:,0]],
                        'Question': ['Q1' for i in range(len(q1))]+['Q2' for i in range(len(q2))]})
         
        chrt = alt.Chart(pca_df).mark_point(filled=True).encode(
            x=alt.X('PC1:Q', axis=alt.Axis(title='PC1')),
            y=alt.Y('PC2:Q', axis=alt.Axis(title='PC2')),
            color=alt.Color('Question:N'),
            tooltip='Word'
        ).properties(
            width=300,
            height=300
        )

        display(chrt)
        
    def explained_variance_plot(self):
        explained_var_df = pd.DataFrame({'Component': np.arange(self.svm_results['s'].shape[0])+1, 
                                         'Explained variance': self.svm_results['s']})
        
        chrt = alt.Chart(explained_var_df).mark_point(size=6).encode(
            alt.X('Component:O'),
            y='Explained variance:Q',
            tooltip=['Component', 'Explained variance']
        ).properties(
            width=600,
            height=200
        )        
        
        display(chrt)
        
    
class QuoraTransformer:
    def __init__(self, embeddings_type=None):
        self.embeddings_type = embeddings_type
        
    def fit(self, df ):
        all_questions = list(df["question1"]) + list(df["question2"])
        self.vectorizer = TfIdfEmbeddingVectorizer(self.embeddings_type).fit(all_questions)
        return self
    
    def transform(self, df):
        q1 = preprocess(df['question1'])
        q2 = preprocess(df['question2'])
                        
        embeddings = [
            self.vectorizer.transform(q1),
            self.vectorizer.transform(q2)
        ]   
                        
        return np.c_[            
            embeddings[0],
            embeddings[1],
            
            cosine(*embeddings),
            manhattan(*embeddings),
            euclid(*embeddings),
            
            jaccard(q1, q2),
            
            word_movers_distance(q1, q2, KeyedVectors.load(self.embeddings_type))
        ]

    
class QuoraBaselineTransformer:
        
    def fit(self, df ):
        all_questions = list(df["question1"]) + list(df["question2"])
        self.vectorizer = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(1,1)).fit(all_questions)               
        
        return self
    
    def transform(self, df):
        return sp.sparse.hstack((
            self.vectorizer.transform(df["question1"]),
            self.vectorizer.transform(df["question2"])
        ))
    
################################################################################
#                               DISTANCES                                      #
################################################################################

def cosine(X, Y):
    norm = (np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1))
    return 1 - np.einsum('ij,ij->i', X, Y) / np.where(norm==0, 1, norm)


def manhattan(X, Y):
    return np.linalg.norm(X - Y, ord=1, axis=1)


def euclid(X, Y):
    return np.linalg.norm(X - Y, ord=2, axis=1)

def jaccard(q1, q2):
    result = np.zeros(len(q1))
    for i in range(len(q1)):
        X, Y = set(q1[i]), set(q2[i])
        result[i] = len(X.intersection(Y)) / max(len(X.union(Y)), 1)
    return result

def word_movers_distance_document_pair(doc1, doc2, model):
    """
    Function used to compute the Word Mover's Distance between two documents.
    """
    doc1_tokens = [token for token in set(doc1) if token in model.key_to_index]
    doc2_tokens = [token for token in set(doc2) if token in model.key_to_index]

    len_tokens_doc1 = len(doc1_tokens)
    len_tokens_doc2 = len(doc2_tokens)

    # If some of the documents is empty, then the required work to move from
    # one to another is 0
    if len_tokens_doc1 == 0 or len_tokens_doc2 == 0:
        return 0

    vocabulary = {
        word: idx
        for idx, word in enumerate(list(set(doc1_tokens) | set(doc2_tokens)))
    }

    doc1_idx = [vocabulary[word] for word in doc1_tokens]
    doc2_idx = [vocabulary[word] for word in doc2_tokens]

    doc1_embeddings = np.array([model[word] for word in doc1_tokens])
    doc2_embeddings = np.array([model[word] for word in doc2_tokens])

    doc1_embeddings = np.repeat(doc1_embeddings, len_tokens_doc2, axis=0)
    doc2_embeddings = np.tile(
        doc2_embeddings.reshape(-1,),
        len_tokens_doc1
    ).reshape(-1, model.vector_size)

    vocabulary_len = len(vocabulary)

    # The distances have to be computed for each pair of embeddings
    # np.ix_ allows to create a mesh to quickly update the necessary entries,
    # but the output has to be properly reshaped
    distances = np.zeros((vocabulary_len, vocabulary_len))
    distances[np.ix_(doc1_idx, doc2_idx)] = euclid(
        doc1_embeddings,
        doc2_embeddings
    ).reshape(
        len_tokens_doc1,
        len_tokens_doc2
    )

    # Get normalized BOW reprsentations of the documents as dense arrays
    bow_doc1 = np.zeros(vocabulary_len)
    bow_doc2 = np.zeros(vocabulary_len)

    data_doc1, ind_col_doc1, _ = bag_of_words([doc1], vocabulary, normalize=True)
    data_doc2, ind_col_doc2, _ = bag_of_words([doc2], vocabulary, normalize=True)

    bow_doc1[ind_col_doc1] = data_doc1
    bow_doc2[ind_col_doc2] = data_doc2

    return emd(bow_doc1, bow_doc2, distances)

def word_movers_distance(X_q1, X_q2, model):
    """
    Function used to compute the Word Mover's Distance between a collection of
    pairs of documents.
    """
    return np.array([
        word_movers_distance_document_pair(doc1, doc2, model)
        for doc1, doc2 in zip(X_q1, X_q2)
    ])

    
