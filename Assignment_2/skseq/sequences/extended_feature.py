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
        feat_name = "id:%s::%s" % (word, y_name)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
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