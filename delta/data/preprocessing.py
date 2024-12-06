import numpy as np
import os


def load_raw_data(root_dir, alignment_strength):
    """
    Load raw data from the alignment tables.
    """

    # For the 'real' data
    raw_data = np.load(os.path.join(root_dir, 'raw_alignment_tables', f"realignments_mu_{alignment_strength}.npz"), allow_pickle=True)
    galaxy_data = raw_data["galaxy_table"]
    orient_data = raw_data["orientation_tables"]

    positions = np.stack([galaxy_data['x'],
                          galaxy_data['y'],
                          galaxy_data['z']], axis=1)

    orientations = np.stack([orient_data[0]["galaxy_axisA_x"],
                             orient_data[0]["galaxy_axisA_y"],
                             orient_data[0]["galaxy_axisA_z"]], axis=1)

    properties = np.stack([(galaxy_data['gal_type'] == 'satellites').astype(int), ], axis=1)

    return positions, orientations, properties


def orientation_to_spin2(orientation_vectors):
    """
    Convert orientation vectors to spin-2 representation.
    """
    norms = np.linalg.norm(orientation_vectors, axis=1, keepdims=True)
    normalized_vectors = orientation_vectors / norms
    theta = np.arctan2(normalized_vectors[:, 1], normalized_vectors[:, 0])
    q_real = np.cos(2 * theta)  # Real part
    q_imag = np.sin(2 * theta)  # Imaginary par
    q_spin2 = np.stack([q_real, q_imag], axis=-1)
    return q_spin2


def preprocessing(positions, orientations, properties):
    """
    Preprocess the data.
    """

    # Positions are already between 0 and 1000. No normalization required.
    positions_processed = positions

    # Orientations are converted to spin-2 representation.
    orientations_processed = orientation_to_spin2(orientations)

    # Properties are already in the correct format, for now.
    properties_processed = properties

    return positions_processed, orientations_processed, properties_processed


def preprocess_data(config):
    """
    Load and preprocess the data, then save it to files. Skip if the files already exist.
    """

    root_dir = config["data"]["data_root"]
    preprocessed_dir = os.path.join(root_dir, 'preprocessed')
    os.makedirs(preprocessed_dir, exist_ok=True)  # Ensure the data root directory exists

    alignment_strength = config["data"]["alignment_strength"]
    alignment_strength = str(alignment_strength)

    # check if the files already exist, if so, skip
    if not os.path.exists(os.path.join(preprocessed_dir, 'positions_{}.npy'.format(alignment_strength))):

        positions, orientations, properties = load_raw_data(root_dir, alignment_strength)
        positions, orientations, properties = preprocessing(positions, orientations, properties)

        # write to files
        np.save(os.path.join(preprocessed_dir, 'positions_{}.npy'.format(alignment_strength)), positions)
        np.save(os.path.join(preprocessed_dir, 'orientations_{}.npy'.format(alignment_strength)), orientations)
        np.save(os.path.join(preprocessed_dir, 'properties_{}.npy'.format(alignment_strength)), properties)

    # Also preprocess 1.0 alignment strength 1.0 if it doesn't exist. This is used for 'truth' comparisons.
    if not os.path.exists(os.path.join(root_dir, 'preprocessed', 'positions_1.0.npy')):

        positions, orientations, properties = load_raw_data(root_dir, alignment_strength="1.0")
        positions, orientations, properties = preprocessing(positions, orientations, properties)

        np.save(os.path.join(preprocessed_dir, 'positions_1.0.npy'), positions)
        np.save(os.path.join(preprocessed_dir, 'orientations_1.0.npy'), orientations)
        np.save(os.path.join(preprocessed_dir, 'properties_1.0.npy'), properties)

    return
