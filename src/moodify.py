import kagglehub

def load_dataset() -> str:
    """
    Downloads the Valorant Champion Tour dataset from Kaggle and returns the path to the dataset files.
    """
    path = kagglehub.dataset_download("msambare/fer2013")
    return path