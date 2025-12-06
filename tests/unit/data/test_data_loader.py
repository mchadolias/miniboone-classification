import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.data import  KaggleDownloader, LocalFileDownloader, DataLoader


class TestDataLoader:
    @mock.patch("pandas.read_csv")
    def test_load_csv_file(self, tmp_path):





class TestLocalFileDownloader:




class TestKaggleDownloader:
