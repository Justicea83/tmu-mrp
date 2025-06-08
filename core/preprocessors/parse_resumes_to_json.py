"""
Resume parsing script that loads resumes from CSV, selects specific categories,
parses the Resume_Str field using AI, and saves the processed data.
"""

import json
import logging
import pandas as pd
from typing import Dict
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to Python path to allow imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from core.openai import QueryEngine
from core.openai.post_processor import PostProcessor
from core.prompt_schema.resume_schema import get_resume_schema
from core.utils import log_message


class ResumeParser:
    """Main class for parsing resumes using AI"""
    
    def __init__(self):
        self.schema = get_resume_schema()
        logging.basicConfig(level=logging.INFO)
        
    def load_resume_data(self, file_path: str) -> pd.DataFrame:
        """
        Load resume data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded resume data
        """
        try:
            log_message(logging.INFO, f"Loading resume data from {file_path}")
            df = pd.read_csv(file_path)
            log_message(logging.INFO, f"Loaded {len(df)} resumes")
            return df
        except Exception as e:
            log_message(logging.ERROR, f"Error loading resume data: {str(e)}", exception=e)
            raise
    
    def select_resumes_by_category(self, df: pd.DataFrame, category_counts: Dict[str, int]) -> pd.DataFrame:
        """
        Select resumes by category with specified counts.
        
        Args:
            df (pd.DataFrame): Full resume dataset
            category_counts (Dict[str, int]): Dictionary of category -> count mappings
            
        Returns:
            pd.DataFrame: Filtered dataset with selected resumes
        """
        selected_resumes = []
        
        for category, count in category_counts.items():
            category_data = df[df['Category'] == category]
            
            if len(category_data) < count:
                log_message(logging.WARNING, 
                          f"Only {len(category_data)} resumes available for {category}, requested {count}")
                selected_count = len(category_data)
            else:
                selected_count = count
            
            # Sample the requested number of resumes from this category
            selected_category_data = category_data.sample(n=selected_count, random_state=42)
            selected_resumes.append(selected_category_data)
            
            log_message(logging.INFO, f"Selected {selected_count} resumes from {category}")
        
        # Combine all selected resumes
        result = pd.concat(selected_resumes, ignore_index=True)
        log_message(logging.INFO, f"Total selected resumes: {len(result)}")
        
        return result
    
    def parse_resume_with_ai(self, resume_text: str) -> Dict:
        """
        Parse a resume using AI to extract structured information.
        
        Args:
            resume_text (str): The text content of the resume.
            
        Returns:
            Dict: The parsed resume data in JSON format.
        """
        try:
            # Prepare the prompt
            prompt_json = json.dumps({
                "resume": resume_text
            }, indent=4)
            
            # Build the query
            query_params = QueryEngine.build_query(
                prompt_json,
                self.schema,
                include_example=False,
                selected_language="English",
                json7schema=True
            )
            
            # Select the model
            model_params = QueryEngine.select_model(query_params)
            
            # Execute the query
            log_message(logging.INFO, "Calling AI to parse resume")
            openai_wrapper = QueryEngine.execute_query(query_params, model_params)
            
            # Get the response
            output = openai_wrapper.first_choice_text
            
            # Convert the response to a dictionary
            parsed_data = PostProcessor.convert_to_dict(output, self.schema)
            
            return parsed_data
            
        except Exception as e:
            log_message(logging.ERROR, f"Error parsing resume with AI: {str(e)}", exception=e)
            return {}
    
    def process_resumes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all resumes in the dataframe by parsing them with AI.
        
        Args:
            df (pd.DataFrame): DataFrame containing resume data
            
        Returns:
            pd.DataFrame: DataFrame with added parsed_json column
        """
        parsed_data = []
        
        for idx, row in df.iterrows():
            log_message(logging.INFO, f"Processing resume {idx + 1}/{len(df)}")
            
            # Parse the resume text
            parsed_json = self.parse_resume_with_ai(row['Resume_str'])
            
            # Create a new row with parsed data
            new_row = row.copy()
            new_row['parsed_json'] = json.dumps(parsed_json) if parsed_json else '{}'
            
            parsed_data.append(new_row)
        
        return pd.DataFrame(parsed_data)
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """
        Save the processed data to CSV.
        
        Args:
            df (pd.DataFrame): Processed DataFrame
            output_path (str): Path to save the CSV file
        """
        try:
            df.to_csv(output_path, index=False)
            log_message(logging.INFO, f"Saved processed data to {output_path}")
        except Exception as e:
            log_message(logging.ERROR, f"Error saving data: {str(e)}", exception=e)
            raise


def main():
    """Main function to run the resume parsing process"""
    
    # Configuration
    DATASET_PATH = "datasets/resumes.csv"
    OUTPUT_PATH = "datasets/resumes_final.csv"
    
    # Categories and counts to select
    CATEGORY_COUNTS = {
        "INFORMATION-TECHNOLOGY": 80,
        "AUTOMOBILE": 20,
        "HR": 20
    }
    
    # Initialize parser
    parser = ResumeParser()
    
    try:
        # Step 1: Load the dataset
        df = parser.load_resume_data(DATASET_PATH)
        
        # Step 2: Select resumes by category
        selected_df = parser.select_resumes_by_category(df, CATEGORY_COUNTS)

        # Step 3: Process resumes with AI
        processed_df = parser.process_resumes(selected_df)
        
        # Step 4: Save the final data
        parser.save_processed_data(processed_df, OUTPUT_PATH)
        
        log_message(logging.INFO, "Resume parsing process completed successfully!")
        
    except Exception as e:
        log_message(logging.ERROR, f"Error in main process: {str(e)}", exception=e)
        raise


if __name__ == "__main__":
    main() 