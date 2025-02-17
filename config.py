from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment("LungLobeFewShotSeg", save_git_info=False)
ex.observers.append(FileStorageObserver('logs'))

@ex.config
def config():
    # Model and dataset paths
    save_model_path = "./save"
    datasets_dir = "./datasets"
    model_path = "save"
    G_path = "save"
    index_m_path = "save"

    # Training parameters
    slicer_len = 2000
    epochs = 30
    max_iter = 500
    record_epochs = 10
    SCCGRD_len = 10
    SCCGRD_alp = 0.98
    
    label_names = ['BG', '1', '2', '3', '4', '5']
    # Optimizer configuration
    seed = 1
    lr = 0.001
    momentum = 0.9
    weight_decay = 0.005

    # Loss function configuration
    loss_weights = [0.1, 1.0]
    ignore_index = 255

    # Learning rate scheduling
    milestones = [1000]
    gamma = 0.95

    # Data loader configuration
    batch_size = 1
    shuffle = False
    num_workers = 2
    pin_memory = True
    drop_last = True

    # File type configuration
    image_file_type = "nii.gz"
    label_file_type = "nii"

    # Save results
    """
    When saving the results, use Lobe_inference and uncomment the related section in inference.py 
    while commenting out the 'testing only' code section.
    """
    save_path = "./"
    save_mhd = True
    save_nii = False