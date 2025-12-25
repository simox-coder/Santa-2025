"""Download competition data from Kaggle."""

import os
import sys
import zipfile
from pathlib import Path


def check_kaggle_credentials():
    """Check if Kaggle credentials are available."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    if kaggle_json.exists():
        return True
    
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    
    return False


def download_competition_data(data_dir: Path = None):
    """Download Santa 2025 competition data."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "raw"
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if not check_kaggle_credentials():
        print("ERROR: Kaggle credentials not found!")
        print()
        print("Please set up Kaggle credentials using ONE of these methods:")
        print()
        print("Method 1: Create ~/.kaggle/kaggle.json with content:")
        print('  {"username": "YOUR_USERNAME", "key": "YOUR_API_KEY"}')
        print()
        print("Method 2: Set environment variables:")
        print("  export KAGGLE_USERNAME=your_username")
        print("  export KAGGLE_KEY=your_api_key")
        print()
        print("Get your API key from: https://www.kaggle.com/account")
        sys.exit(1)
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        print(f"Downloading competition data to {data_dir}...")
        api.competition_download_files("santa-2025", path=str(data_dir))
        
        # Unzip if needed
        zip_files = list(data_dir.glob("*.zip"))
        for zip_file in zip_files:
            print(f"Extracting {zip_file}...")
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extractall(data_dir)
            zip_file.unlink()  # Remove zip after extraction
        
        print("Download complete!")
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    download_competition_data()


if __name__ == "__main__":
    main()
