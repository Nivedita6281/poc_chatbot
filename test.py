# save this as clear_index.py
import os
import shutil
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_faiss_index():
    """
    Clears the FAISS index by deleting index files with improved error handling.
    """
    # Path to the FAISS index
    index_path = "faiss_index"
    
    # Check if directory exists
    if not os.path.exists(index_path):
        logger.warning(f"Index directory {index_path} does not exist")
        return True  # Nothing to delete
    
    # Try to delete individual files instead of the whole directory
    try:
        for filename in os.listdir(index_path):
            file_path = os.path.join(index_path, filename)
            
            # Try multiple times with small delays
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        logger.info(f"Deleted file: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        logger.info(f"Deleted directory: {file_path}")
                    # If we get here, the operation was successful
                    break
                except PermissionError:
                    if attempt < max_attempts - 1:
                        logger.info(f"Permission error, retrying in 2 seconds... (attempt {attempt+1}/{max_attempts})")
                        time.sleep(2)  # Wait before retrying
                    else:
                        logger.error(f"Failed to delete {file_path} after {max_attempts} attempts")
                        return False
        
        # Create an empty index.pkl file to indicate an empty index
        with open(os.path.join(index_path, "index.pkl"), "w") as f:
            f.write("")
        
        return True
    except Exception as e:
        logger.error(f"Failed to clear vector store: {e}")
        return False

if __name__ == "__main__":
    print("Attempting to clear FAISS index...")
    print("Make sure your FastAPI application is not running!")
    
    input("Press Enter to continue...")
    
    success = clear_faiss_index()
    if success:
        print("Vector store cleared successfully. Restart your application for changes to take effect.")
    else:
        print("Failed to clear vector store. Check the logs for details.")
        print("\nAlternative method: manually delete the faiss_index folder")
        print("1. Close all applications that might be using these files")
        print("2. Open File Explorer and navigate to your project directory")
        print("3. Delete the faiss_index folder")
        print("4. Restart your application")