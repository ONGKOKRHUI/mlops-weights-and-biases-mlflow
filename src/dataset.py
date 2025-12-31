import torch
from torch.utils.data import Dataset
import sqlite3
import numpy as np
import io

class MNISTDatabaseDataset(Dataset):
    def __init__(self, db_path, split='train'):
        self.db_path = db_path
        self.split = split
        
        # Pre-fetch indices to allow __len__ and __getitem__ to work
        # We assume the DB exists. If not, run mnist_to_sqlite.py first.
        self.indices = self._get_indices()

    def _get_indices(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM mnist_data WHERE split=?", (self.split,))
        indices = [row[0] for row in cursor.fetchall()]
        conn.close()
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Open a fresh connection for thread safety if num_workers > 0
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        real_id = self.indices[idx]
        cursor.execute("SELECT image, label FROM mnist_data WHERE id=?", (real_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result is None:
            raise IndexError(f"ID {real_id} not found in database.")

        img_bytes, label = result
        
        # Deserialize numpy array from bytes
        with io.BytesIO(img_bytes) as bio:
            img_array = np.load(bio)
            
        # Convert to Tensor: (1, 28, 28) float32 normalized to [0, 1]
        image = torch.from_numpy(img_array).float() / 255.0
        image = image.unsqueeze(0) 

        return image, label