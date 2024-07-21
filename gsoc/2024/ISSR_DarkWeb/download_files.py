import os
import gdown
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PATH_CONTENT = "Analyze_files/CombiningAnalysisCompleteDataset/ContentAnalysis"
PATH_THREAD = "Analyze_files/CombiningAnalysisCompleteDataset/ThreadAnalysis/Models"


def download_file_from_google_drive(file_id: str, destination: str) -> None:
    '''
    Downloads a file from Google Drive using the file ID and saves it to the specified destination.
    :param file_id: The ID of the file on Google Drive.
    :param destination: The path where the downloaded file should be saved.
    '''
    url = f"https://drive.google.com/uc?id={file_id}"

    # Use gdown to download the file to the specified destination
    gdown.download(url, destination, quiet=False)


def add_copy_suffix(destination: str) -> str:
    """
    Adds '_copy' before the file extension in the given destination path.
    :param destination: The destination path where the file will be copied.
    :return: The modified destination path with '_copy' added before the file extension.
    """
    base, ext = os.path.splitext(destination)
    return f"{base}_copy{ext}"



# List of files to download with their respective Google Drive file IDs and destination paths
files_to_download = [
    {
        "file_id": "1mIHndJj5VBNIkmGzKzIzFnniKm4sQ6Bq",
        "destination": f"{PATH_CONTENT}/DatasetsContentBERTopic/BERTopic_all-MiniLM-L6-v2_190_20n_8dim.parquet",
        "file_size": "444 MB"
    },
    {
        "file_id": "1odJkRR78sUyb-GWruXbwESNVdcnl1Li-",
        "destination": f"{PATH_CONTENT}/PreProcessFiles/content_preprocessed_embeddings.npz",
        "file_size": "360 MB"
    },
    {
        "file_id": "1hfvDRtmwJdqISwmxT_pSp1fQS3_HftcS",
        "destination": f"{PATH_CONTENT}/ModelsContent/topic_model_all-MiniLM-L6-v2_190_20n_8dim_safetensors/ctfidf_config.json",
        "file_size": "360 MB"
    },
    {
        "file_id": "1tgqwoKuTb_f53fSvM0YgghE0jrfpNsbV",
        "destination": f"{PATH_CONTENT}/ModelsContent/topic_model_all-MiniLM-L6-v2_190_20n_8dim_safetensors/ctfidf_config.safetensors",
        "file_size": "230 MB"
    },
    {
        "file_id": "1svXmaMaVVN_-ifqnaKtangtyYvFM_mGo",
        "destination": f"{PATH_CONTENT}/ModelsContent/topic_model_all-MiniLM-L6-v2_190_20n_8dim",
        "file_size": "2.5 GB"
    },
    {
        "file_id": "1M-f-T7URHM457bpbP5sX0qjPUf9ve4ON",
        "destination": f"{PATH_THREAD}/topic_model_0.64SilNew",
        "file_size": "560 MB"
    },
    {
        "file_id": "1otWPdqFcoXIcQDfm0_ozGDtyHvMuhsxu",
        "destination": f"{PATH_THREAD}/topic_model_0.50Sil300",
        "file_size": "560 MB"
    },
    {
        "file_id": "1laGx-5mp29QNs58EtbThbqepNneiPc1r",
        "destination": f"{PATH_THREAD}/topic_model_all-MiniLM-L6-v2_150_20n",
        "file_size": "550 MB"
    },
    {
        "file_id": "1MoWnRbJUE49DgdvcmMzQNKz_zwylqQXj",
        "destination": f"{PATH_THREAD}/topic_model_all-MiniLM-L6-v2_400",
        "file_size": "520 MB"
    },
    {
        "file_id": "1607IlsZljEyCEvdYu_5BxDrDoHMZ-KSq",
        "destination": f"{PATH_THREAD}/topic_model_all-MiniLM-L6-v2_200",
        "file_size": "520 MB"
    }
]



for file in files_to_download:
    file["destination"] = add_copy_suffix(file["destination"])


# Download each file listed in files_to_download
for file in tqdm(files_to_download, desc="Downloading files"):
    while True:
        file_name = os.path.basename(file['destination'])
        should_download = input(f"\nDo you want to download the file {file_name} ({file['file_size']})? (yes/no or y/n): ").strip().lower()
        
        if should_download in ['yes', 'y']:
            # Create the directory structure if it doesn't exist
            os.makedirs(os.path.dirname(file["destination"]), exist_ok=True)

            # Download the file using the defined function
            try:
                download_file_from_google_drive(file["file_id"], file["destination"])
                logging.info(f"Download completed. The file has been saved to: {file['destination']}")
            except Exception as e:
                logging.error(f"An error occurred while downloading the file: {e}")
            break
        elif should_download in ['no', 'n']:
            logging.info(f"Skipped downloading the file to: {file['destination']}")
            break
        else:
            logging.warning("Invalid input. Please enter 'yes', 'no', 'y', or 'n'.")

logging.info("All download operations have been completed.")