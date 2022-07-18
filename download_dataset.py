"""
Downloads the csv data
"""

import logging
import os
import shutil

import pandas as pd
import urllib3

# Initial dataset source
DATASET_URL = "http://bit.ly/building-ml-pipelines-dataset"

# Initial local dataset location
LOCAL_FILE_NAME = "data/consumer_complaints_with_narrative.csv"


def download_dataset(url=DATASET_URL):
    """download_dataset downloads the remote dataset to a local path
    Keyword Arguments:
        url {string} --
            complete url path to the csv data source (default: {DATASET_URL})
        local_path {string} --
            initial local file location (default: {LOCAL_FILE_NAME})
    Returns:
        None
    """
    # disable insecure https warning
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    c = urllib3.PoolManager()
    with c.request("GET", url, preload_content=False) as res, open(
        LOCAL_FILE_NAME, "wb"
    ) as out_file:
        shutil.copyfileobj(res, out_file)
    logging.info("Download completed.")


def create_folder():
    """Creates a data folder if it doesn't exist.
    Returns:
        None
    """
    directory = "data/"
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info("Data folder created.")
    else:
        logging.info("Data folder already existed.")

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.info("Started download script")

    create_folder()
    download_dataset()

    logging.info("Finished download script")