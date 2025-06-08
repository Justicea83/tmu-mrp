#!/usr/bin/env python3
"""
Resume-Job Ranking Runner

This script runs the resume-job matching and ranking system on a specified number 
of jobs and resumes from CSV files.

Usage:
    python runners/rank.py --num-jobs 10 --num-resumes 50
    python runners/rank.py --jobs-file custom_jobs.csv --resumes-file custom_resumes.csv --num-jobs 5 --num-resumes 20
"""

import argparse
import pandas as pd
import sys
import os
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.models import Resume, JobDescription, ResumeJobMatch
from core.matching_engine.engine import compute_resume_job_match
from core.explainable_ai import ExplainableAI
from core.diversity_analytics import DiversityAnalytics
from core.learning_to_rank import LearningToRankEngine

# Create logs directory if it doesn't exist
LOGS_DIR = 'logs'
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, f'ranking_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def ensure_logs_directory(filepath: str) -> str:
    """
    Ensure the filepath is in the logs directory.
    If no directory is specified, prepend logs directory.
    
    Args:
        filepath: Original file path
        
    Returns:
        File path ensured to be in logs directory
    """
    if not os.path.dirname(filepath):
        return os.path.join(LOGS_DIR, filepath)
    return filepath


def load_resumes_from_csv(file_path: str, num_resumes: Optional[int] = None,
                          categories: Optional[List[str]] = None,
                          exclude_categories: Optional[List[str]] = None,
                          balanced_categories: bool = True) -> List[Resume]:
    """
    Load resumes from CSV file with optional category filtering and balanced sampling.
    
    Args:
        file_path: Path to the resumes CSV file
        num_resumes: Number of resumes to load (None for all)
        categories: List of categories to include (None for all)
        exclude_categories: List of categories to exclude (None for none)
        balanced_categories: If True and categories specified, try to sample equally from each category
        
    Returns:
        List of Resume objects
    """
    logger.info(f"Loading resumes from {file_path}")

    try:
        df = pd.read_csv(file_path)

        # Apply category filtering
        if categories:
            df = df[df['Category'].isin(categories)]
            logger.info(f"Filtered to categories: {categories}")

        if exclude_categories:
            df = df[~df['Category'].isin(exclude_categories)]
            logger.info(f"Excluded categories: {exclude_categories}")

        # Balanced sampling across categories if requested
        if num_resumes is not None and categories and balanced_categories and len(categories) > 1:
            # Calculate resumes per category for balanced sampling
            resumes_per_category = num_resumes // len(categories)
            remainder = num_resumes % len(categories)

            logger.info(f"Attempting balanced sampling: {resumes_per_category} resumes per category")
            if remainder > 0:
                logger.info(f"Will add {remainder} extra resume(s) to first {remainder} category(ies)")

            balanced_dfs = []
            category_counts = {}

            for i, category in enumerate(categories):
                category_df = df[df['Category'] == category]
                available_count = len(category_df)

                # Calculate target count for this category
                target_count = resumes_per_category + (1 if i < remainder else 0)
                actual_count = min(target_count, available_count)

                if actual_count > 0:
                    # Sample from this category
                    sampled_df = category_df.head(actual_count)
                    balanced_dfs.append(sampled_df)
                    category_counts[category] = actual_count

                if available_count < target_count:
                    logger.warning(
                        f"Category '{category}' has only {available_count} resumes, requested {target_count}")

            if balanced_dfs:
                df = pd.concat(balanced_dfs, ignore_index=True)
                logger.info(f"Balanced sampling results: {category_counts}")
            else:
                logger.warning("No resumes found for balanced sampling, falling back to regular sampling")
                df = df.head(num_resumes)
        elif num_resumes is not None:
            df = df.head(num_resumes)

        resumes = []
        for _, row in df.iterrows():
            try:
                # Create Resume object from CSV row
                resume = Resume(
                    ID=str(row.get('ID', '')),
                    Resume_str=str(row.get('Resume_str', '')),
                    Resume_html=str(row.get('Resume_html', '')),
                    Category=str(row.get('Category', '')),
                    hash=str(row.get('hash', '')),
                    char_len=str(row.get('char_len', '')),
                    sent_len=str(row.get('sent_len', '')),
                    type_token_ratio=str(row.get('type_token_ratio', '')),
                    gender_term_count=str(row.get('gender_term_count', '')),
                    html_len=str(row.get('html_len', '')),
                    text_from_html=str(row.get('text_from_html', '')),
                    parsed_json=str(row.get('parsed_json', '')),
                    html_strip_diff=str(row.get('html_strip_diff', ''))
                )
                resumes.append(resume)

            except Exception as e:
                logger.warning(f"Failed to create resume from row {row.name}: {e}")
                continue

        logger.info(f"Successfully loaded {len(resumes)} resumes")
        return resumes

    except Exception as e:
        logger.error(f"Failed to load resumes from {file_path}: {e}")
        raise


def load_jobs_from_csv(file_path: str, num_jobs: Optional[int] = None,
                       job_keywords: Optional[List[str]] = None) -> List[JobDescription]:
    """
    Load job descriptions from CSV file with optional keyword filtering.
    
    Args:
        file_path: Path to the job descriptions CSV file
        num_jobs: Number of jobs to load (None for all)
        job_keywords: List of keywords to filter jobs by position (None for all)
        
    Returns:
        List of JobDescription objects
    """
    logger.info(f"Loading job descriptions from {file_path}")

    try:
        df = pd.read_csv(file_path)

        # Apply keyword filtering
        if job_keywords:
            # Create a mask for jobs that contain any of the keywords in their position
            keyword_mask = df['Position'].str.contains('|'.join(job_keywords), case=False, na=False)
            df = df[keyword_mask]
            logger.info(f"Filtered jobs by keywords: {job_keywords}")
            logger.info(f"Found {len(df)} jobs matching keywords")

        if num_jobs is not None:
            df = df.head(num_jobs)

        jobs = []
        for _, row in df.iterrows():
            try:
                # Create JobDescription object from CSV row
                job = JobDescription(
                    Position=str(row.get('Position', '')),
                    Long_Description=str(row.get('Long Description', '')),
                    Company_Name=str(row.get('Company Name', '')),
                    Exp_Years=str(row.get('Exp Years', '')),
                    Primary_Keyword=str(row.get('Primary Keyword', '')),
                    English_Level=str(row.get('English Level', '')),
                    Published=str(row.get('Published', '')),
                    Long_Description_lang=str(row.get('Long Description_lang', '')),
                    id=str(row.get('id', '')),
                    __index_level_0__=str(row.get('__index_level_0__', '')),
                    char_len=str(row.get('char_len', ''))
                )
                jobs.append(job)

            except Exception as e:
                logger.warning(f"Failed to create job description from row {row.name}: {e}")
                continue

        logger.info(f"Successfully loaded {len(jobs)} job descriptions")
        return jobs

    except Exception as e:
        logger.error(f"Failed to load job descriptions from {file_path}: {e}")
        raise


def run_ranking(resumes: List[Resume], jobs: List[JobDescription],
                top_k: int = 10, verbose: bool = False,
                weights: Optional[Dict[str, float]] = None,
                general_model: str = None, skills_model: str = None) -> List[Dict[str, Any]]:
    """
    Run the ranking system on the provided resumes and jobs.
    
    Args:
        resumes: List of Resume objects
        jobs: List of JobDescription objects
        top_k: Number of top matches to return per job
        verbose: Enable verbose logging
        weights: Optional dictionary mapping matcher names to weights for overall score calculation
        general_model: The sentence transformer model to use for general matching
        skills_model: The sentence transformer model to use for skills matching
        
    Returns:
        List of ranking results
    """
    logger.info(f"Starting ranking process for {len(resumes)} resumes and {len(jobs)} jobs")

    results = []
    total_combinations = len(jobs) * len(resumes)
    processed = 0

    for job_idx, job in enumerate(jobs):
        logger.info(f"Processing job {job_idx + 1}/{len(jobs)}: {job.Position}")

        job_matches = []

        for resume_idx, resume in enumerate(resumes):
            try:
                # Create ResumeJobMatch object with unique match_id
                match_id = f"job_{job.id}_resume_{resume.ID}_{job_idx}_{resume_idx}"
                match = ResumeJobMatch(resume=resume, job_description=job, match_id=match_id)

                # Compute the match score
                match_data = compute_resume_job_match(match, compute_overall=True, weights=weights,
                                                      general_model=general_model, skills_model=skills_model)

                # Extract total score and individual scores
                overall_data = match_data.get('overall', {})
                total_score = overall_data.get('score', 0.0) if isinstance(overall_data, dict) else 0.0

                score_breakdown = {
                    'general': match_data.get('general', {}).get('score', 0.0) if isinstance(match_data.get('general'),
                                                                                             dict) else 0.0,
                    'skills': match_data.get('skills', {}).get('score', 0.0) if isinstance(match_data.get('skills'),
                                                                                           dict) else 0.0,
                    'experience': match_data.get('experience', {}).get('score', 0.0) if isinstance(
                        match_data.get('experience'), dict) else 0.0,
                    'location': match_data.get('location', {}).get('score', 0.0) if isinstance(
                        match_data.get('location'), dict) else 0.0,
                    'education': match_data.get('education', {}).get('score', 0.0) if isinstance(
                        match_data.get('education'), dict) else 0.0
                }

                job_matches.append({
                    'resume_id': resume.ID,
                    'resume_category': resume.Category,
                    'total_score': total_score,
                    'score_breakdown': score_breakdown,
                    'resume_char_len': resume.char_len,
                    'resume': resume
                })

                processed += 1
                if processed % 100 == 0:
                    logger.info(
                        f"Processed {processed}/{total_combinations} combinations ({processed / total_combinations * 100:.1f}%)")

            except Exception as e:
                logger.warning(f"Failed to process match between resume {resume.ID} and job {job.id}: {e}")
                continue

        # Sort matches by total score (descending)
        job_matches.sort(key=lambda x: x['total_score'], reverse=True)

        # Take top k matches for results output
        top_matches = job_matches[:top_k]

        results.append({
            'job_id': job.id,
            'job_position': job.Position,
            'job_company': job.Company_Name,
            'total_candidates': len(job_matches),
            'top_matches': top_matches,
            'all_matches': job_matches  # Store ALL matches for comprehensive analysis
        })

        # Log top matches for this job
        logger.info(f"Top {min(top_k, len(top_matches))} matches for job '{job.Position}':")
        for i, match in enumerate(top_matches[:5], 1):  # Show top 5 in logs
            logger.info(
                f"  {i}. Resume {match['resume_id']} (Category: {match['resume_category']}) - Score: {match['total_score']:.2f}")

    logger.info(f"Ranking complete. Processed {processed} total combinations.")
    return results


def save_results(results: List[Dict[str, Any]], output_file: str, category_analysis: bool = False):
    """
    Save ranking results to a CSV file with optional category analysis.
    
    Args:
        results: List of ranking results
        output_file: Path to output CSV file
        category_analysis: Whether to include detailed category analysis
    """
    logger.info(f"Saving results to {output_file}")

    # Flatten results for CSV export (top matches only)
    flattened_results = []

    for job_result in results:
        job_id = job_result['job_id']
        job_position = job_result['job_position']
        job_company = job_result['job_company']

        for rank, match in enumerate(job_result['top_matches'], 1):
            flattened_results.append({
                'job_id': job_id,
                'job_position': job_position,
                'job_company': job_company,
                'rank': rank,
                'resume_id': match['resume_id'],
                'resume_category': match['resume_category'],
                'total_score': match['total_score'],
                'general_score': match['score_breakdown'].get('general', 0),
                'skills_score': match['score_breakdown'].get('skills', 0),
                'experience_score': match['score_breakdown'].get('experience', 0),
                'location_score': match['score_breakdown'].get('location', 0),
                'education_score': match['score_breakdown'].get('education', 0),
                'resume_char_len': match['resume_char_len']
            })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(flattened_results)
    df.to_csv(output_file, index=False)

    logger.info(f"Results saved to {output_file}")
    logger.info(f"Total matches saved: {len(flattened_results)}")

    # Perform category analysis if requested
    if category_analysis and results:
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE CATEGORY ANALYSIS")
        logger.info("=" * 60)

        # Create analysis DataFrame from ALL processed matches (not just top-k)
        all_matches_data = []
        for job_result in results:
            job_id = job_result['job_id']
            job_position = job_result['job_position']
            job_company = job_result['job_company']

            # Use all_matches for comprehensive analysis
            for match in job_result.get('all_matches', []):
                all_matches_data.append({
                    'job_id': job_id,
                    'job_position': job_position,
                    'job_company': job_company,
                    'resume_id': match['resume_id'],
                    'resume_category': match['resume_category'],
                    'total_score': match['total_score'],
                    'general_score': match['score_breakdown'].get('general', 0),
                    'skills_score': match['score_breakdown'].get('skills', 0),
                    'experience_score': match['score_breakdown'].get('experience', 0),
                    'location_score': match['score_breakdown'].get('location', 0),
                    'education_score': match['score_breakdown'].get('education', 0),
                    'resume_char_len': match['resume_char_len']
                })

        # Create comprehensive analysis DataFrame
        all_df = pd.DataFrame(all_matches_data)

        if not all_df.empty:
            logger.info(f"Analyzing ALL {len(all_df)} processed resume-job combinations")

            # Analyze performance by resume category (ALL processed resumes)
            category_stats = all_df.groupby('resume_category').agg({
                'total_score': ['mean', 'std', 'count', 'min', 'max'],
                'general_score': 'mean',
                'skills_score': 'mean',
                'experience_score': 'mean',
                'location_score': 'mean',
                'education_score': 'mean'
            }).round(2)

            logger.info("Performance by Resume Category (ALL PROCESSED RESUMES):")
            for category in category_stats.index:
                stats = category_stats.loc[category]
                logger.info(f"  {category}:")
                logger.info(f"    - Total Processed: {stats[('total_score', 'count')]}")
                logger.info(
                    f"    - Avg Total Score: {stats[('total_score', 'mean')]} ± {stats[('total_score', 'std')]}")
                logger.info(f"    - Score Range: {stats[('total_score', 'min')]} - {stats[('total_score', 'max')]}")
                logger.info(f"    - Avg General: {stats[('general_score', 'mean')]}")
                logger.info(f"    - Avg Skills: {stats[('skills_score', 'mean')]}")
                logger.info(f"    - Avg Experience: {stats[('experience_score', 'mean')]}")
                logger.info(f"    - Avg Location: {stats[('location_score', 'mean')]}")
                logger.info(f"    - Avg Education: {stats[('education_score', 'mean')]}")

            # Show top performers from each category
            logger.info("\nTop Performers by Category:")
            for category in all_df['resume_category'].unique():
                category_matches = all_df[all_df['resume_category'] == category]
                top_match = category_matches.loc[category_matches['total_score'].idxmax()]
                logger.info(f"  {category}: Resume {top_match['resume_id']} - Score: {top_match['total_score']:.2f}")

            # Analyze performance by job position (with comprehensive category distribution)
            job_stats = all_df.groupby('job_position').agg({
                'total_score': ['mean', 'std', 'count'],
                'resume_category': lambda x: x.value_counts().to_dict()
            }).round(2)

            logger.info("\nPerformance by Job Position (ALL PROCESSED):")
            for job_pos in job_stats.index:
                stats = job_stats.loc[job_pos]
                logger.info(f"  {job_pos}:")
                logger.info(f"    - Total Processed: {stats[('total_score', 'count')]}")
                logger.info(f"    - Avg Score: {stats[('total_score', 'mean')]} ± {stats[('total_score', 'std')]}")
                logger.info(f"    - Category Distribution: {stats[('resume_category', '<lambda>')]}")

            # Show category representation in top-k vs all processed
            top_df = pd.DataFrame(flattened_results)
            if not top_df.empty:
                logger.info(f"\nCategory Representation Analysis:")
                logger.info(f"Top-{len(top_df)} Results vs All {len(all_df)} Processed:")

                for category in all_df['resume_category'].unique():
                    total_processed = len(all_df[all_df['resume_category'] == category])
                    in_top_k = len(top_df[top_df['resume_category'] == category])
                    percentage = (in_top_k / total_processed * 100) if total_processed > 0 else 0

                    logger.info(f"  {category}:")
                    logger.info(f"    - Processed: {total_processed}, In Top-K: {in_top_k} ({percentage:.1f}%)")

            # Save comprehensive analysis to separate file
            analysis_file = output_file.replace('.csv', '_comprehensive_analysis.csv')
            # Ensure analysis file is in logs directory if output_file doesn't already have a directory
            if not os.path.dirname(analysis_file):
                analysis_file = os.path.join(LOGS_DIR, analysis_file)
            category_stats.to_csv(analysis_file)
            logger.info(f"Comprehensive category analysis saved to: {analysis_file}")

            # Also save all processed results for detailed analysis
            all_results_file = output_file.replace('.csv', '_all_processed.csv')
            if not os.path.dirname(all_results_file):
                all_results_file = os.path.join(LOGS_DIR, all_results_file)
            all_df.to_csv(all_results_file, index=False)
            logger.info(f"All processed results saved to: {all_results_file}")
        else:
            logger.warning("No comprehensive analysis data available")


def run_model_comparison(resumes: List[Resume], jobs: List[JobDescription],
                         models_to_compare: List[str], top_k: int = 10,
                         weights: Optional[Dict[str, float]] = None,
                         output_base: str = "model_comparison") -> Dict[str, List[Dict[str, Any]]]:
    """
    Run ranking with multiple models for comparison.
    
    Args:
        resumes: List of Resume objects
        jobs: List of JobDescription objects
        models_to_compare: List of model names to compare
        top_k: Number of top matches to return per job
        weights: Optional dictionary mapping matcher names to weights
        output_base: Base name for output files
        
    Returns:
        Dictionary mapping model names to their ranking results
    """
    logger.info("=" * 60)
    logger.info("MODEL COMPARISON MODE")
    logger.info("=" * 60)
    logger.info(f"Comparing models: {models_to_compare}")

    comparison_results = {}
    all_model_data = []

    for model_name in models_to_compare:
        logger.info(f"\n🔄 Running with model: {model_name}")
        logger.info("-" * 40)

        # Run ranking with this model for both general and skills
        results = run_ranking(resumes, jobs, top_k, verbose=False, weights=weights,
                              general_model=model_name, skills_model=model_name)

        comparison_results[model_name] = results

        # Collect data for comparison analysis
        for job_result in results:
            job_id = job_result['job_id']
            job_position = job_result['job_position']

            for rank, match in enumerate(job_result['top_matches'], 1):
                all_model_data.append({
                    'model': model_name,
                    'job_id': job_id,
                    'job_position': job_position,
                    'rank': rank,
                    'resume_id': match['resume_id'],
                    'resume_category': match['resume_category'],
                    'total_score': match['total_score'],
                    'general_score': match['score_breakdown'].get('general', 0),
                    'skills_score': match['score_breakdown'].get('skills', 0),
                    'experience_score': match['score_breakdown'].get('experience', 0),
                    'location_score': match['score_breakdown'].get('location', 0),
                    'resume_char_len': match['resume_char_len']
                })

        # Save individual model results
        model_output_file = f"{output_base}_{model_name.replace('/', '_').replace('-', '_')}.csv"
        model_output_file = ensure_logs_directory(model_output_file)
        save_results(results, model_output_file, category_analysis=True)

        # Print summary for this model
        total_matches = sum(len(job_result['top_matches']) for job_result in results)
        avg_score = sum(match['total_score'] for job_result in results
                        for match in job_result['top_matches']) / total_matches if total_matches > 0 else 0

        logger.info(f"✅ {model_name} Results:")
        logger.info(f"   - Average Score: {avg_score:.2f}")
        logger.info(f"   - Total Matches: {total_matches}")
        logger.info(f"   - Results saved to: {model_output_file}")

    # Create comparison analysis
    logger.info("\n📊 GENERATING COMPARISON ANALYSIS")
    logger.info("-" * 40)

    comparison_df = pd.DataFrame(all_model_data)

    # Model performance comparison
    model_stats = comparison_df.groupby('model').agg({
        'total_score': ['mean', 'std', 'count'],
        'general_score': 'mean',
        'skills_score': 'mean',
        'experience_score': 'mean',
        'location_score': 'mean'
    }).round(3)

    logger.info("Model Performance Comparison:")
    for model in model_stats.index:
        stats = model_stats.loc[model]
        logger.info(f"  {model}:")
        logger.info(f"    - Avg Total Score: {stats[('total_score', 'mean')]} ± {stats[('total_score', 'std')]}")
        logger.info(f"    - Avg General: {stats[('general_score', 'mean')]}")
        logger.info(f"    - Avg Skills: {stats[('skills_score', 'mean')]}")
        logger.info(f"    - Avg Experience: {stats[('experience_score', 'mean')]}")
        logger.info(f"    - Avg Location: {stats[('location_score', 'mean')]}")

    # Save comparison data
    comparison_output_file = ensure_logs_directory(f"{output_base}_detailed_comparison.csv")
    comparison_df.to_csv(comparison_output_file, index=False)

    model_stats_file = ensure_logs_directory(f"{output_base}_model_statistics.csv")
    model_stats.to_csv(model_stats_file)

    logger.info(f"\n📁 Comparison Files Generated:")
    logger.info(f"   - Detailed comparison: {comparison_output_file}")
    logger.info(f"   - Model statistics: {model_stats_file}")

    return comparison_results


def main():
    parser = argparse.ArgumentParser(description='Run resume-job ranking system')

    # File arguments
    parser.add_argument('--resumes-file', type=str, default='datasets/resumes_final.csv',
                        help='Path to resumes CSV file (default: datasets/resumes_final.csv)')
    parser.add_argument('--jobs-file', type=str, default='datasets/job_descriptions.csv',
                        help='Path to job descriptions CSV file (default: datasets/job_descriptions.csv)')

    # Quantity arguments
    parser.add_argument('--num-resumes', type=int, default=1,
                        help='Number of resumes to process (default: 100)')
    parser.add_argument('--num-jobs', type=int, default=1,
                        help='Number of job descriptions to process (default: 10)')

    # Output arguments
    parser.add_argument('--output-file', type=str,
                        default=os.path.join(LOGS_DIR,
                                             f'ranking_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'),
                        help='Path to output CSV file')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of top matches to return per job (default: 10)')

    # Logging arguments
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    # Weight configuration arguments
    parser.add_argument('--general-weight', type=float, default=1.0,
                        help='Weight for general matcher (default: 1.0)')
    parser.add_argument('--skills-weight', type=float, default=1.0,
                        help='Weight for skills matcher (default: 1.0)')
    parser.add_argument('--experience-weight', type=float, default=1.0,
                        help='Weight for experience matcher (default: 1.0)')
    parser.add_argument('--location-weight', type=float, default=1.0,
                        help='Weight for location matcher (default: 1.0)')
    parser.add_argument('--education-weight', type=float, default=1.0,
                        help='Weight for education matcher (default: 1.0)')

    # Sentence transformer model arguments
    parser.add_argument('--general-model', type=str, default=None,
                        help='Sentence transformer model for general matching (default: careerbert)')
    parser.add_argument('--skills-model', type=str, default=None,
                        help='Sentence transformer model for skills matching (default: careerbert)')
    parser.add_argument('--model-comparison', action='store_true',
                        help='Run comparison with multiple models (careerbert vs all-mpnet-base-v2)')
    parser.add_argument('--models-to-compare', nargs='+',
                        default=['careerbert', 'sentence-transformers/all-mpnet-base-v2'],
                        help='List of models to compare (default: careerbert, all-mpnet-base-v2)')

    # Category filtering arguments
    parser.add_argument('--resume-categories', nargs='+',
                        help='Filter by resume categories (e.g., INFORMATION-TECHNOLOGY AUTOMOBILE HR)')
    parser.add_argument('--job-keywords', nargs='+',
                        help='Filter jobs by keywords in position (e.g., developer engineer manager)')
    parser.add_argument('--exclude-resume-categories', nargs='+',
                        help='Exclude specific resume categories')
    parser.add_argument('--category-analysis', action='store_true',
                        help='Include detailed category analysis in results')
    parser.add_argument('--balanced-categories', action='store_true', default=True,
                        help='Balance resume selection across categories when multiple categories specified (default: True)')
    parser.add_argument('--no-balanced-categories', dest='balanced_categories', action='store_false',
                        help='Disable balanced category sampling')

    # Advanced features arguments
    parser.add_argument('--explainable-ai', action='store_true',
                        help='Generate detailed explanations for rankings')
    parser.add_argument('--diversity-analysis', action='store_true',
                        help='Perform diversity and bias analysis')
    parser.add_argument('--learning-to-rank', action='store_true',
                        help='Use machine learning to improve rankings')
    parser.add_argument('--ltr-model-type', type=str, default='gradient_boosting',
                        choices=['linear', 'random_forest', 'gradient_boosting'],
                        help='Learning-to-rank model type (default: gradient_boosting)')

    args = parser.parse_args()

    # Ensure output file is in logs directory
    args.output_file = ensure_logs_directory(args.output_file)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("RESUME-JOB RANKING SYSTEM")
    logger.info("=" * 60)
    logger.info(f"Resumes file: {args.resumes_file}")
    logger.info(f"Jobs file: {args.jobs_file}")
    logger.info(f"Number of resumes: {args.num_resumes}")
    logger.info(f"Number of jobs: {args.num_jobs}")
    logger.info(f"Top-k matches per job: {args.top_k}")
    logger.info(f"Output file: {args.output_file}")

    # Create weights dictionary
    raw_weights = {
        'general': args.general_weight,
        'skills': args.skills_weight,
        'experience': args.experience_weight,
        'location': args.location_weight,
        'education': args.education_weight
    }

    # Normalize weights to sum to 1.0
    total_weight = sum(raw_weights.values())
    if total_weight == 0:
        logger.error("Total weight cannot be zero!")
        sys.exit(1)

    weights = {key: weight / total_weight for key, weight in raw_weights.items()}

    logger.info(f"Raw matcher weights: {raw_weights}")
    logger.info(f"Normalized weights (sum=1.0): {weights}")
    logger.info(
        f"Weight distribution: General={weights['general']:.3f}, Skills={weights['skills']:.3f}, Experience={weights['experience']:.3f}, Location={weights['location']:.3f}, Education={weights['education']:.3f}")

    # Log model configuration
    if args.model_comparison:
        logger.info(f"Model comparison mode: ENABLED")
        logger.info(f"Models to compare: {args.models_to_compare}")
    else:
        logger.info(f"General model: {args.general_model or 'careerbert (default)'}")
        logger.info(f"Skills model: {args.skills_model or 'careerbert (default)'}")

    # Log filtering options
    if args.resume_categories:
        logger.info(f"Resume categories filter: {args.resume_categories}")
        logger.info(f"Balanced category sampling: {'ENABLED' if args.balanced_categories else 'DISABLED'}")
    if args.exclude_resume_categories:
        logger.info(f"Excluded resume categories: {args.exclude_resume_categories}")
    if args.job_keywords:
        logger.info(f"Job keywords filter: {args.job_keywords}")
    if args.category_analysis:
        logger.info("Category analysis: ENABLED")

    logger.info("=" * 60)

    try:
        # Validate input files
        if not os.path.exists(args.resumes_file):
            raise FileNotFoundError(f"Resumes file not found: {args.resumes_file}")
        if not os.path.exists(args.jobs_file):
            raise FileNotFoundError(f"Jobs file not found: {args.jobs_file}")

        # Load data with category filtering
        resumes = load_resumes_from_csv(
            args.resumes_file,
            args.num_resumes,
            categories=args.resume_categories,
            exclude_categories=args.exclude_resume_categories,
            balanced_categories=args.balanced_categories
        )
        jobs = load_jobs_from_csv(
            args.jobs_file,
            args.num_jobs,
            job_keywords=args.job_keywords
        )

        if not resumes:
            raise ValueError("No resumes loaded")
        if not jobs:
            raise ValueError("No jobs loaded")

        # Check if model comparison mode is enabled
        if args.model_comparison:
            # Run model comparison
            comparison_results = run_model_comparison(
                resumes, jobs, args.models_to_compare, args.top_k, weights,
                output_base=os.path.splitext(os.path.basename(args.output_file))[0]
            )

            logger.info("=" * 60)
            logger.info("MODEL COMPARISON COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)

            # Print comparison summary
            logger.info("Model Comparison Summary:")
            for model_name, results in comparison_results.items():
                total_matches = sum(len(job_result['top_matches']) for job_result in results)
                avg_score = sum(match['total_score'] for job_result in results
                                for match in job_result['top_matches']) / total_matches if total_matches > 0 else 0
                logger.info(f"  {model_name}: Avg Score = {avg_score:.2f}, Matches = {total_matches}")

        else:
            # Run standard ranking with specified models
            results = run_ranking(resumes, jobs, args.top_k, args.verbose, weights,
                                  general_model=args.general_model, skills_model=args.skills_model)

            # Save results
            save_results(results, args.output_file, args.category_analysis)

            # Run advanced features if requested
            if args.explainable_ai or args.diversity_analysis or args.learning_to_rank:
                logger.info("=" * 60)
                logger.info("RUNNING ADVANCED FEATURES")
                logger.info("=" * 60)

                # Flatten all results for advanced analysis
                all_results = []
                for job_result in results:
                    for rank, match in enumerate(job_result['top_matches'], 1):
                        result_entry = {
                            'job_id': job_result['job_id'],
                            'job_position': job_result['job_position'],
                            'job_company': job_result['job_company'],
                            'rank': rank,
                            'resume_id': match['resume_id'],
                            'resume_category': match['resume_category'],
                            'total_score': match['total_score'],
                            'general_score': match['score_breakdown'].get('general', 0),
                            'skills_score': match['score_breakdown'].get('skills', 0),
                            'experience_score': match['score_breakdown'].get('experience', 0),
                            'location_score': match['score_breakdown'].get('location', 0),
                            'education_score': match['score_breakdown'].get('education', 0)
                        }
                        all_results.append(result_entry)

                # Explainable AI
                if args.explainable_ai:
                    logger.info("Generating explainable AI analysis...")
                    explainer = ExplainableAI()
                    
                    # Generate explanations for top matches
                    explanations = []
                    for i, result in enumerate(all_results[:10]):  # Top 10 for detailed explanation
                        # Find corresponding resume and job
                        resume = next((r for r in resumes if r.ID == result['resume_id']), None)
                        job = next((j for j in jobs if str(j.id) == str(result['job_id'])), None)
                        
                        if resume and job:
                            scores = {
                                'general': result['general_score'],
                                'skills': result['skills_score'],
                                'experience': result['experience_score'],
                                'location': result['location_score'],
                                'education': result['education_score']
                            }
                            
                            explanation = explainer.explain_ranking(
                                resume.__dict__, job.__dict__, scores, weights, all_results
                            )
                            explanations.append({
                                'rank': result['rank'],
                                'resume_id': result['resume_id'],
                                'job_position': result['job_position'],
                                'explanation': explanation
                            })
                    
                    # Save explanations
                    explanation_file = ensure_logs_directory(f"explanations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                    import json
                    import numpy as np
                    
                    # Custom JSON encoder to handle numpy types
                    class NumpyEncoder(json.JSONEncoder):
                        def default(self, obj):
                            if isinstance(obj, np.integer):
                                return int(obj)
                            elif isinstance(obj, np.floating):
                                return float(obj)
                            elif isinstance(obj, np.ndarray):
                                return obj.tolist()
                            elif isinstance(obj, np.bool_):
                                return bool(obj)
                            return super(NumpyEncoder, self).default(obj)
                    
                    with open(explanation_file, 'w') as f:
                        json.dump(explanations, f, indent=2, cls=NumpyEncoder)
                    logger.info(f"Explanations saved to: {explanation_file}")

                # Diversity Analysis
                if args.diversity_analysis:
                    logger.info("Performing diversity and bias analysis...")
                    diversity_analyzer = DiversityAnalytics()
                    
                    # Convert resumes to dict format for analysis
                    resumes_dict = [resume.__dict__ for resume in resumes]
                    
                    diversity_analysis = diversity_analyzer.analyze_diversity_metrics(all_results, resumes_dict)
                    
                    # Save diversity analysis
                    diversity_file = ensure_logs_directory(f"diversity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                    with open(diversity_file, 'w') as f:
                        json.dump(diversity_analysis, f, indent=2, cls=NumpyEncoder)
                    logger.info(f"Diversity analysis saved to: {diversity_file}")
                    
                    # Generate and save bias report
                    bias_report = diversity_analyzer.generate_bias_report(all_results, resumes_dict)
                    bias_report_file = ensure_logs_directory(f"bias_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                    with open(bias_report_file, 'w') as f:
                        f.write(bias_report)
                    logger.info(f"Bias report saved to: {bias_report_file}")

                # Learning to Rank
                if args.learning_to_rank:
                    logger.info(f"Training learning-to-rank model ({args.ltr_model_type})...")
                    ltr_engine = LearningToRankEngine(model_type=args.ltr_model_type)
                    
                    try:
                        # Convert data for LTR
                        resumes_dict = [resume.__dict__ for resume in resumes]
                        jobs_dict = [job.__dict__ for job in jobs]
                        
                        # Prepare training data
                        features, labels = ltr_engine.prepare_training_data(all_results, resumes_dict, jobs_dict)
                        
                        # Train model
                        training_results = ltr_engine.train_model(features, labels)
                        logger.info(f"Model training completed. Validation MSE: {training_results['val_mse']:.4f}")
                        
                        # Re-rank using ML model
                        ml_results = ltr_engine.rank_candidates(all_results, resumes_dict, jobs_dict)
                        
                        # Save ML results
                        ml_results_file = ensure_logs_directory(f"ml_ranking_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                        ml_df = pd.DataFrame(ml_results)
                        ml_df.to_csv(ml_results_file, index=False)
                        logger.info(f"ML ranking results saved to: {ml_results_file}")
                        
                        # Save model
                        model_file = ensure_logs_directory(f"ltr_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
                        ltr_engine.save_model(model_file)
                        logger.info(f"Trained model saved to: {model_file}")
                        
                        # Generate feature importance report
                        importance_report = ltr_engine.get_feature_importance_report()
                        importance_file = ensure_logs_directory(f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                        with open(importance_file, 'w') as f:
                            f.write(importance_report)
                        logger.info(f"Feature importance report saved to: {importance_file}")
                        
                        # Evaluate ranking quality
                        evaluation = ltr_engine.evaluate_ranking_quality(all_results, ml_results)
                        logger.info(f"Ranking correlation: {evaluation['ranking_correlation']:.3f}")
                        logger.info(f"Score improvement: {evaluation['score_improvement']['improvement']:.3f}")
                        
                    except Exception as e:
                        logger.error(f"Learning-to-rank failed: {e}")

            logger.info("=" * 60)
            logger.info("RANKING COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)

            # Print summary statistics
            total_matches = sum(len(job_result['top_matches']) for job_result in results)
            avg_score = sum(match['total_score'] for job_result in results
                            for match in job_result['top_matches']) / total_matches if total_matches > 0 else 0

            logger.info(f"Summary:")
            logger.info(f"  - Jobs processed: {len(results)}")
            logger.info(f"  - Total top matches: {total_matches}")
            logger.info(f"  - Average match score: {avg_score:.2f}")
            logger.info(f"  - Results saved to: {args.output_file}")



    except Exception as e:
        logger.error(f"Ranking failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
