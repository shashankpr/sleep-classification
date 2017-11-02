from src import RunLSTM, DataGenerator

file_path = "src/hdf/master_file_all_subjects.hdf"

#TODO Add other LSTM parameters as and when needed
def run(file_path, n_features, seq_len, n_workers):
    """Main function to run the individual classes

    Args:
        file_path: Path to the HDF file
        n_features (int) : Number of features to consider : 7 for HRV and 2 for raw signal.
        seq_len    (int) : Specify the sequence_length to be used.
        n_workers  (int) : Number of threads needed to load HDF data using generators.

    Returns: Confusion matrix of test samples

    """

    # d = DataGenerator(file_path=file_path)
    # d.gen_sample_count()
    lstm_model = RunLSTM(file_path=file_path, SEQ_LEN=seq_len, FEATURES_DIM=n_features, nb_workers=n_workers)
    lstm_model.run_gen_model()

if __name__ == '__main__':
    run(file_path=file_path, n_features = 7, seq_len = 8, n_workers = 2)


