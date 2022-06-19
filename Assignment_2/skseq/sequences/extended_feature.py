import re

from skseq.sequences.id_feature import IDFeatures
from skseq.sequences.id_feature import UnicodeFeatures

# ----------
# Feature Class
# Extracts features from a labeled corpus (only supported features are extracted
# ----------
class ExtendedFeatures(IDFeatures):

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
        # Generate feature name.
        feat_name = f"id:{word}::{y_name}"
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
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
        initialism_pattern = re.compile(r"(\w\.){2,}")

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


        # Feature: is floating point number
        try:
            float(word)

            feat_name = f"float::{y_name}"
            feat_id = self.add(feat_name)

            if feat_id != -1:
                features.append(feat_id)
        except ValueError:
            pass


        # Feature: word contains dot
        if len(word) > 1 and '.' in word:
            feat_name = f"has_dot::{y_name}"
            feat_id = self.add(feat_name)

            if feat_id != -1:
                features.append(feat_id)


        return features

# eg prefix features: get the three first characters of the word
# suffic: last features
# weather there are hypens in the words
# weather there are floating points numbers in the sequence
# etc
    
class ExtendedFeatures(IDFeatures):    
    def noseonva():
        if str.isdigit(word):
            #generate feature name
            feat_name = "number::%s" %y_name
            feat_name = str(feat_name)
            
            #get feature id from
            #...