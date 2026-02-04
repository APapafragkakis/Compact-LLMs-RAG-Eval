
from langchain_community.document_loaders.pdf import PyPDFLoader
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging
from langchain_community.document_loaders.csv_loader import CSVLoader
import csv
from typing import List


from KG_RAG.utils import benchmark

class DataLoader:
    def __init__(self, csv_args=None):
        self.csv_args = csv_args or {}

        self.loaders = {
            "pdf": PyPDFLoader,
            "csv": lambda file_path: load_csv_with_auto_header(file_path)
        }

    @benchmark
    def load(self, dir_path: str, extensions: List[str] = []):
        dir_path = Path(dir_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Provided path '{dir_path}' is not a valid directory.")

        if len(extensions) == 0:
            files = {ext: list(dir_path.glob(f"*.{ext}")) for ext in self.loaders}
        else:
            files = {ext: list(dir_path.glob(f"*.{ext}")) for ext in extensions}
        files_to_load = [(file, ext) for ext, file_list in files.items() for file in file_list]

        if not files_to_load:
            logging.warning("No supported files found in the directory.")
            return []

        documents = []

        def load_file(file_info):
            file_path, ext = file_info
            logging.info(f"Loading {file_path.name}")
            return self.loaders[ext](str(file_path)).load()

        with ThreadPoolExecutor() as executor:
            results = executor.map(load_file, files_to_load)

        for doc in results:
            documents.extend(doc)

        return documents


# Todo: Automatically detect if header exists
def load_csv_with_auto_header(file_path):
    csv_args = {}
    has_header = True
    if (not has_header):
        with open(file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            first_row = next(reader)
            num_columns = len(first_row)
            fieldnames = [f"c_{i}" for i in range(num_columns)]
            csv_args = {'fieldnames': fieldnames}
            
    logging.info(f"Loading {file_path} (Detected Header: {has_header})")
    return CSVLoader(file_path, csv_args=csv_args)