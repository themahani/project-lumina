import os
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi


def download_and_unzip_data():
    """
    Connects to the Kaggle API to download and unzip the Elliptic dataset.
    """
    # Configuration
    dataset = "ellipticco/elliptic-data-set"
    download_path = "data/raw"

    # Ensure the target directory exists
    os.makedirs(download_path, exist_ok=True)

    # Check if data already exists to avoid re-downloading
    expected_files = [
        "elliptic_txs_classes.csv",
        "elliptic_txs_edgelist.csv",
        "elliptic_txs_features.csv",
    ]
    if all(os.path.exists(os.path.join(download_path, f)) for f in expected_files):
        print("Data files already exist. Skipping download.")
        return

    print(f"Downloading dataset '{dataset}' from Kaggle to '{download_path}'...")

    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download the dataset
    api.dataset_download_files(dataset, path=download_path, quiet=False)

    print("Download complete. Unzipping files...")

    zip_path = os.path.join(download_path, "elliptic-data-set.zip")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(download_path)

    # Clean up the downloaded zip file
    os.remove(zip_path)

    print(f"Data successfully unzipped to '{download_path}'.")


if __name__ == "__main__":
    try:
        download_and_unzip_data()
    except ImportError:
        print(
            "Kaggle package not found. Please install it by running: pip install kaggle"
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        print(
            "Please ensure your kaggle.json API token is correctly placed in ~/.kaggle/ or C:\\Users\\<Your-Username>\\.kaggle"
        )
