import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class DataLoader:
    """
    Class to handle data loading.
    """
    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):
        try:
            df = pd.read_csv(self.filepath)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found at {self.filepath}")
            raise