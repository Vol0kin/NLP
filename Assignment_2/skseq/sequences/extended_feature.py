import re
import joblib
import gensim.downloader as api

from skseq.sequences.id_feature import IDFeatures
from skseq.sequences.id_feature import UnicodeFeatures

# ----------
# Feature Class
# Extracts features from a labeled corpus (only supported features are extracted
# ----------
class ExtendedFeatures(IDFeatures):
    def __init__(self, dataset, model_path):
        super().__init__(dataset)

        self.word2vec = api.load('word2vec-google-news-300')
        self.k_means = joblib.load(model_path)


    def add_emission_features(self, sequence, pos, y, features):
        x = sequence.x[pos]
        # Get tag name from ID.
        y_name = self.dataset.y_dict.get_label_name(y)
        # Get word name from ID.
        if isinstance(x, str):
            x_name = x
        else:
            x_name = self.dataset.x_dict.get_label_name(x)
        word = str(x_name)
        
        # Feature: HMM-like emission features
        feat_name = f"id:{word}::{y_name}" # Generate feature name.
        feat_id = self.add_feature(feat_name) # Get feature ID from name.
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)

        # Feature: first letter is capitalized
        if word.istitle():
            feat_name = f"init_caps::{y_name}"
            feat_id = self.add_feature(feat_name) 

            if feat_id != -1:
                features.append(feat_id)


        # Feature: all letters are uppercase
        if word.isupper():
            feat_name = f"all_caps::{y_name}"
            feat_id = self.add_feature(feat_name)

            if feat_id != -1:
                features.append(feat_id)


        # Feature: Initialism
        initialism_pattern = re.compile(r"([a-zA-Z]\.){2,}")

        if bool(initialism_pattern.match(word)):
            feat_name = f"initialism::{y_name}"
            feat_id = self.add_feature(feat_name)

            if feat_id != -1:
                features.append(feat_id)


        # Feature: is digit
        if str.isdigit(word):
            feat_name = f"digit::{y_name}"
            feat_id = self.add_feature(feat_name)

            if feat_id != -1:
                features.append(feat_id)
            
            # Features: digits of length 1, 2 or 4 
            for i in [1, 2, 4]:
                if len(word) == i:
                    feat_name = f"digit:{str(i)}::{y_name}"
                    feat_id = self.add_feature(feat_name)
                    if feat_id != -1:
                        features.append(feat_id)        


        # Feature: is floating point number
        try:
            float(word)

            feat_name = f"float::{y_name}"
            feat_id = self.add_feature(feat_name)

            if feat_id != -1:
                features.append(feat_id)
        except ValueError:
            pass


        # Feature: word contains dot
        if len(word) > 1 and '.' in word:
            feat_name = f"has_dot::{y_name}"
            feat_id = self.add_feature(feat_name)

            if feat_id != -1:
                features.append(feat_id)


        # Feature: word cluster
        try:
            embedding = self.word2vec[word]
            cluster = self.k_means.predict(embedding.reshape(1, -1))[0]
        except KeyError:
            cluster = len(self.k_means.labels_)
        
        feat_name = f"cluster:{cluster}::{y_name}"
        feat_id = self.add_feature(feat_name)

        if feat_id != -1:
            features.append(feat_id)


        return features
