import os
import shutil

from sentence_transformers import SentenceTransformer

# Original model
APPLICATION_SENTENCE_TRANSFORMER_MODEL = 'sentence-transformers/all-mpnet-base-v2'
MODEL_DIR = './sentence-transformers/all-mpnet-base-v2'

# New model for job matching
CAREER_BERT_MODEL = 'lwolfrum2/careerbert-jg'
CAREER_BERT_MODEL_DIR = './sentence-transformers/careerbert-jg'


def download_model(model_name, model_dir):
    """
    Download a sentence transformer model and save it to the specified directory.

    Args:
        model_name: The name of the model to download.
        model_dir: The directory to save the model to.
    """
    # Check if model is already properly downloaded by looking for key files
    model_exists = False
    if os.path.exists(model_dir):
        # Check for essential model files
        required_files = ['config.json']
        model_files = ['pytorch_model.bin', 'model.safetensors']
        
        has_config = any(os.path.exists(os.path.join(model_dir, f)) for f in required_files)
        has_model = any(os.path.exists(os.path.join(model_dir, f)) for f in model_files)
        
        model_exists = has_config and has_model
    
    if model_exists:
        print(f"Model {model_name} already downloaded and appears complete.")
    else:
        print(f"Downloading model {model_name}...")
        
        # Clean up any incomplete downloads
        if os.path.exists(model_dir):
            print(f"Removing incomplete download at {model_dir}")
            shutil.rmtree(model_dir)
        
        try:
            # Download the model from HuggingFace
            model = SentenceTransformer(model_name)
            # Now create the directory and save
            os.makedirs(model_dir, exist_ok=True)
            model.save(model_dir)
            print(f"Model {model_name} downloaded and saved to {model_dir}")
        except Exception as e:
            print(f"Error downloading model {model_name}: {str(e)}")
            # Clean up failed download
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            raise


def download():
    """
    Download all required sentence transformer models.
    """
    # Download the original model
    download_model(APPLICATION_SENTENCE_TRANSFORMER_MODEL, MODEL_DIR)

    # Download the career bert model
    download_model(CAREER_BERT_MODEL, CAREER_BERT_MODEL_DIR)


if __name__ == '__main__':
    download()
