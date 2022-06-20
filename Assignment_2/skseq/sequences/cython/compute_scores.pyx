import skseq.sequences.sequence_classifier as sc
import numpy as np
cimport numpy as np
cimport cython
np.import_array()

def compute_scores(sp, sequence):
    cdef int num_states = sp.get_num_states()
    cdef int length = len(sequence.x)
    cdef double[:,:] emission_scores = np.zeros([length, num_states])
    cdef double[:] initial_scores = np.zeros(num_states)
    cdef double[:,:,:] transition_scores = np.zeros([length-1, num_states, num_states])
    cdef double[:] final_scores = np.zeros(num_states)

    cdef double score
    cdef int feat_id, pos, tag_id, prev_tag_id
    cdef double[:] parameters = sp.parameters

    # Initial position.
    for tag_id in range(num_states):
        initial_features = sp.feature_mapper.get_initial_features(sequence, tag_id)
        score = 0.0
        for feat_id in initial_features:
            score += sp.parameters[feat_id]
        initial_scores[tag_id] = score

    # Intermediate position.
    for pos in range(length):
        for tag_id in range(num_states):
            emission_features = sp.feature_mapper.get_emission_features(sequence, pos, tag_id)
            score = 0.0
            for feat_id in emission_features:
                score += sp.parameters[feat_id]
            emission_scores[pos, tag_id] = score
        if pos > 0:
            for tag_id in range(num_states):
                for prev_tag_id in range(num_states):
                    transition_features = sp.feature_mapper.get_transition_features(
                        sequence, pos, tag_id, prev_tag_id)
                    score = 0.0
                    for feat_id in transition_features:
                        score += sp.parameters[feat_id]
                    transition_scores[pos-1, tag_id, prev_tag_id] = score

    # Final position.
    for prev_tag_id in range(num_states):
        final_features = sp.feature_mapper.get_final_features(sequence, prev_tag_id)
        score = 0.0
        for feat_id in final_features:
            score += sp.parameters[feat_id]
        final_scores[prev_tag_id] = score

    return np.asarray(initial_scores), np.asarray(transition_scores),\
        np.asarray(final_scores), np.asarray(emission_scores)
