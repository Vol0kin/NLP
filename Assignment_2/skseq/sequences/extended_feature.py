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
    def __init__(self, dataset, model_path, default_cluster):
        super().__init__(dataset)

        self.word2cluster = joblib.load(model_path)
        self.default_cluster = default_cluster


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

        # HMM-LIKE EMISSION FEATURES
        feat_name = f"id:{word}::{y_name}" # Generate feature name.
        feat_id = self.add_feature(feat_name) # Get feature ID from name.
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)

        # CONTEXTUAL FEATURES
        # Words surrounding Xi
        # For capitalized words we get a broader context
        if word.istitle():
            positions = [-3, -2, -1, 1, 2, 3]
        # For the other words we look at previous and next word
        else:
            positions = [-1, 1]

        for i_ctx in positions:
            i = pos + i_ctx
            if (i >= 0) & (i < len(sequence)):
                x_ctx = sequence.x[i]
                if not isinstance(x_ctx, str):
                    x_ctx = str(self.dataset.x_dict.get_label_name(x_ctx))
                feat_name = f"context{str(i_ctx)}:{x_ctx}::{y_name}" 
                feat_id = self.add_feature(feat_name)

                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)

        # EXTERNAL INFORMATION
        # English most common affixes
        # suffixes, prefixes = get_affixes() # crashes due to too many requests during training
        suffixes = ['able', 'ity', 'en', 'cy', 'less', 'ry', 'ment', 'er', 'ful', 'sion', 'ence', 'ism', 'ness', 'ery', 'ent', 'ant', 'fy', 'ous','ship', 'ate', 'ise', 'al', 'ive', 'tion', 'age', 'ance']
        prefixes = ['sur', 'anti', 'out', 'tri', 'neo', 'under', 'vice', 'over', 'mini', 'bi', 'counter', 'mono', 'auto', 'super', 'be', 'hyper', 'ir', 'inter', 'il', 'mis', 'im', 'sub', 'mal', 'non', 'un', 'ex', 'co', 'kilo', 'semi', 'pseudo', 'trans', 'dis', 'mega', 're', 'ultra', 'fore', 'poly', 'de', 'pre', 'tele', 'in']

        # Suffixes
        for suffix in suffixes:
            regx = re.compile(r'\w+'+suffix+'$')     
            if bool(regx.match(word)):
                feat_name = f"suffix:{suffix}::{y_name}"
                feat_id = self.add_feature(feat_name)
                if feat_id != -1:
                    features.append(feat_id)    

        for prefix in prefixes:
            regx = re.compile(r'^'+prefix)     
            if bool(regx.match(word)):
                feat_name = f"prefix:{prefix}::{y_name}"
                feat_id = self.add_feature(feat_name)
                if feat_id != -1:
                    features.append(feat_id)    

        # WORD STRUCTURE FEATURES
        # Feature: first letter is capitalized
        if word.istitle():
            feat_name = f"uppercased:first::{y_name}"
            feat_id = self.add_feature(feat_name) 

            if feat_id != -1:
                features.append(feat_id)

        # Feature: all letters are uppercase
        if word.isupper():
            feat_name = f"uppercased:all::{y_name}"
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

        # Features: word length
        for i in range(1, 11):
            if len(word) == i:
                feat_name = f"length:{str(i)}::{y_name}"
                feat_id = self.add_feature(feat_name)

                if feat_id != -1:
                    features.append(feat_id)

        if len(word) > 10 & len(word) <= 15:
            feat_name = f"length:10-15::{y_name}"
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        if len(word) > 15:
            feat_name = f"length:gt15::{y_name}"
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id) 

        # Features: detect several patterns
        patterns = [r"(\w\.){2,}", r"\d+s", r"\d+st", r"\d+nd", r"\d+rd", r"\d+th", r"\-", r"(^\'\w+\'$)|(^\"\w+\"$)", "([A-Z]+[a-z]+)|([a-z]+[A-Z]+)", r"\'"]
        labels = ["initialism","digit:s","digit:st","digit:nd","digit:rd","digit:th","hypend","quote","uppercased:mixed","apostrophe"]

        for pattern, label in zip(patterns, labels):
            regx = re.compile(pattern)

            if bool(regx.match(word)):
                feat_name = f"{label}::{y_name}"
                feat_id = self.add_feature(feat_name)
                if feat_id != -1:
                    features.append(feat_id)    


        # Feature: word cluster
        try:
            cluster = self.word2cluster[word]
        except KeyError:
            cluster = self.default_cluster
        
        feat_name = f"cluster:{cluster}::{y_name}"
        feat_id = self.add_feature(feat_name)

        if feat_id != -1:
            features.append(feat_id)


        return features
