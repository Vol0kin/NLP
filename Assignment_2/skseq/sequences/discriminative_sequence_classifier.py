import skseq.sequences.sequence_classifier as sc
import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
import skseq.sequences.cython.compute_scores as compute_scores

class DiscriminativeSequenceClassifier(sc.SequenceClassifier):

    def __init__(self, observation_labels, state_labels, feature_mapper):
        sc.SequenceClassifier.__init__(self, observation_labels, state_labels)

        # Set feature mapper and initialize parameters.
        self.feature_mapper = feature_mapper
        self.parameters = np.zeros(self.feature_mapper.get_num_features())

    # ----------
    #  Build the node and edge potentials
    # node - f(t,y_t,X)*w
    # edge - f(t,y_t,y_(t-1),X)*w
    # Only supports binary features representation
    # If we have an HMM with 4 positions and transitins
    # a - b - c - d
    # the edge potentials have at position:
    # 0 a - b
    # 1 b - c
    # ----------
    def compute_scores(self, sequence):
        return compute_scores.compute_scores(self, sequence)
