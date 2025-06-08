#!/usr/bin/env python3
"""
Example: Integrating Hugging Face Datasets with Education Matcher

This script demonstrates how to enhance the EducationMatcher with additional
field mappings from Hugging Face datasets for even more comprehensive
education-job matching.

Requirements:
    pip install datasets transformers

Usage:
    python examples/education_dataset_integration.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.matching_engine.education import EducationMatcher
from core.models import Resume, JobDescription, ResumeJobMatch
import logging

def demo_huggingface_integration():
    """Demonstrate integration with Hugging Face datasets."""
    
    print("🎓 Education Matcher - Default Hugging Face Integration Demo")
    print("=" * 65)
    
    # Create test data (simplified for demo - normally loaded from CSV)
    print("\n   🔥 NEW: The system now AUTOMATICALLY loads from Hugging Face!")
    print("      - meliascosta/wiki_academic_subjects (64k academic subjects)")
    print("      - Automatic fallback to comprehensive local mappings")
    print("\n   💡 Note: In production, resume/job data is loaded from CSV files")
    print("      This demo shows the automatic Hugging Face integration")
    
    # For demo purposes, we'll simulate the education matcher without full objects
    # In practice, the matcher gets initialized from the ranking system
    
    # Initialize education matcher with default comprehensive mappings
    print("\n1. Default Comprehensive Mappings:")
    print("-" * 40)
    
    # Create a temporary matcher instance to show field mappings
    temp_matcher = EducationMatcher.__new__(EducationMatcher)
    temp_matcher.degree_levels = {}
    temp_matcher.field_mappings = {}
    temp_matcher.academic_subjects = {}
    temp_matcher.field_keywords = {}
    temp_matcher._load_field_mappings()
    
    print(f"   Field categories: {len(temp_matcher.field_mappings)}")
    print(f"   Total field keywords: {sum(len(fields) for fields in temp_matcher.field_mappings.values())}")
    print(f"   Technology fields: {len(temp_matcher.field_mappings.get('technology', []))}")
    
    # Show some example mappings
    print(f"\n   Sample technology fields:")
    for field in temp_matcher.field_mappings.get('technology', [])[:5]:
        print(f"     - {field}")
    
    print(f"\n   Education field categories available:")
    for category in sorted(temp_matcher.field_mappings.keys()):
        field_count = len(temp_matcher.field_mappings[category])
        print(f"     - {category}: {field_count} fields")
    
    print("\n2. Available Hugging Face Datasets:")
    print("-" * 40)
    datasets_info = [
        {
            'name': 'meliascosta/wiki_academic_subjects',
            'description': '64k academic subject hierarchies from Wikipedia',
            'size': '64.4k rows',
            'use_case': 'Academic subject classification'
        },
        {
            'name': 'jacob-hugging-face/job-descriptions', 
            'description': '853 job descriptions with field classifications',
            'size': '853 rows',
            'use_case': 'Job field mapping'
        },
        {
            'name': 'millawell/wikipedia_field_of_science',
            'description': '304k scientific field taxonomies with hierarchical labels',
            'size': '304k rows',
            'structure': 'token (list), label (hierarchical list like ["Humanities", "Philosophy", "Social philosophy"])',
            'use_case': 'Scientific field classification and academic subject mapping'
        }
    ]
    
    for i, dataset in enumerate(datasets_info, 1):
        print(f"\n   {i}. {dataset['name']}")
        print(f"      Description: {dataset['description']}")
        print(f"      Size: {dataset['size']}")
        if 'structure' in dataset:
            print(f"      Structure: {dataset['structure']}")
        print(f"      Use case: {dataset['use_case']}")
    
    print("\n3. Integration Examples:")
    print("-" * 40)
    print("   # Install required libraries:")
    print("   pip install datasets transformers")
    print()
    print("   # Example 1: Using the default wiki_academic_subjects (automatic)")
    print("""
   from core.matching_engine.education import EducationMatcher
   
   # Automatically loads meliascosta/wiki_academic_subjects by default
   matcher = EducationMatcher(match)
   print(f"Loaded {len(matcher.field_mappings)} field categories")
   """)
    
    print("   # Example 2: Adding scientific field dataset")
    print("""
   # Load additional scientific field taxonomies
   matcher.update_field_mappings_from_dataset("millawell/wikipedia_field_of_science")
   
   # This dataset has structure: 
   # - token: list of tokens
   # - label: hierarchical list like ["Humanities", "Philosophy", "Social philosophy"]
   print(f"Enhanced with scientific fields: {len(matcher.field_mappings)} categories")
   """)
    
    print("   # Example 3: Manual dataset processing")
    print("""
   from datasets import load_dataset
   
   # Load scientific dataset manually
   dataset = load_dataset("millawell/wikipedia_field_of_science", split="train")
   
   for example in dataset.take(5):  # Show first 5 examples
       tokens = example.get('token', [])
       labels = example.get('label', [])
       print(f"Tokens: {tokens[:3]}...")  # First 3 tokens
       print(f"Labels: {labels}")         # Full label hierarchy
   """)
    
    print("\n4. Current System Benefits:")
    print("-" * 40)
    print("   ✅ 177 comprehensive field mappings already included")
    print("   ✅ 10 major field categories covered")
    print("   ✅ Technology, Business, Science, Healthcare, etc.")
    print("   ✅ Enhanced keyword matching with specialized terms")
    print("   ✅ Degree level hierarchy with proper scoring")
    print("   ✅ Experience-education alignment algorithms")
    
    print("\n5. When to Use External Datasets:")
    print("-" * 40)
    print("   📊 Need extremely specialized field mappings")
    print("   📊 Working with non-English education systems")
    print("   📊 Require real-time dataset updates")
    print("   📊 Need domain-specific taxonomies")
    print("   📊 Working with academic research applications")
    
    print("\n6. Implementation Notes:")
    print("-" * 40)
    print("   💡 Current mappings cover most common use cases")
    print("   💡 External datasets add complexity and dependencies")
    print("   💡 Consider caching processed dataset results")
    print("   💡 Validate dataset quality before integration")
    print("   💡 Monitor performance impact of large datasets")
    
    print(f"\n🎯 With these comprehensive mappings, the education matcher")
    print("   can effectively score resume-job education alignment!")
    print("\n✨ The built-in mappings provide excellent coverage for")
    print("   most resume-job matching scenarios!")

def load_academic_subjects_example():
    """Example of how to load and process the academic subjects dataset."""
    
    print("\n" + "=" * 60)
    print("📚 Academic Subjects Dataset Example")
    print("=" * 60)
    
    # This is how you would load the dataset if the datasets library is available
    example_code = '''
from datasets import load_dataset

def enhance_education_matcher_with_academic_subjects(matcher):
    """
    Enhance education matcher with academic subjects from Wikipedia.
    
    Args:
        matcher: EducationMatcher instance to enhance
    """
    try:
        # Load academic subjects dataset
        dataset = load_dataset("meliascosta/wiki_academic_subjects", split="train")
        
        print(f"Loaded {len(dataset)} academic subject examples")
        
        # Process examples to extract field mappings
        new_mappings = {}
        
        for example in dataset:
            token_sequence = example.get('token sequence', [])
            label_sequence = example.get('label sequence', [])
            
            if len(label_sequence) >= 1:
                # Extract hierarchical labels
                if len(label_sequence) >= 2:
                    broad_category = label_sequence[0].lower().replace(' ', '_')
                    specific_field = label_sequence[-1].lower()
                    
                    # Clean and normalize field names
                    specific_field = specific_field.replace('_', ' ')
                    
                    if broad_category not in new_mappings:
                        new_mappings[broad_category] = set()
                    
                    new_mappings[broad_category].add(specific_field)
        
        # Convert sets to lists and merge with existing mappings
        for category, fields in new_mappings.items():
            fields_list = list(fields)
            
            if category in matcher.field_mappings:
                # Merge with existing, avoiding duplicates
                existing = set(matcher.field_mappings[category])
                new_fields = [f for f in fields_list if f not in existing]
                matcher.field_mappings[category].extend(new_fields)
            else:
                # Add new category
                matcher.field_mappings[category] = fields_list
        
        print(f"Enhanced matcher with {len(new_mappings)} field categories")
        print(f"Total fields now: {sum(len(f) for f in matcher.field_mappings.values())}")
        
        return matcher
        
    except ImportError:
        print("datasets library not installed. Install with: pip install datasets")
        return matcher
    except Exception as e:
        print(f"Error loading academic subjects: {e}")
        return matcher

# Usage example:
# matcher = EducationMatcher(match)
# enhanced_matcher = enhance_education_matcher_with_academic_subjects(matcher)
'''
    
    print("Example implementation:")
    print(example_code)

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run demo
    demo_huggingface_integration()
    load_academic_subjects_example()
    
    print("\n" + "=" * 60)
    print("🚀 Demo completed! The education matcher now has comprehensive")
    print("   field mappings and can be enhanced with external datasets.")
    print("=" * 60) 