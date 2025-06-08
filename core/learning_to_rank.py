"""
Learning-to-Rank Module for Resume-Job Ranking System

This module implements advanced ranking models that can learn from feedback
and improve ranking performance over time.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Callable
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import ndcg_score, mean_squared_error
import logging
import pickle
import os


class LearningToRankEngine:
    """
    Advanced learning-to-rank engine for resume-job matching.
    
    Features:
    - Multiple ranking algorithms (pointwise, pairwise, listwise)
    - Feature engineering from existing matchers
    - Feedback incorporation
    - Model performance evaluation
    - Incremental learning capabilities
    """
    
    def __init__(self, model_type: str = 'gradient_boosting'):
        self.logger = logging.getLogger(__name__)
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Model options
        self.model_options = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Feature importance tracking
        self.feature_importance = {}
        self.feature_names = []
        
        # Performance metrics
        self.training_history = []
        self.evaluation_metrics = {}

    def extract_features(self, resume: Dict[str, Any], job: Dict[str, Any], 
                        scores: Dict[str, float]) -> np.ndarray:
        """
        Extract comprehensive features for ranking model.
        
        Args:
            resume: Resume data
            job: Job description data  
            scores: Individual matcher scores
            
        Returns:
            Feature vector as numpy array
        """
        
        features = []
        
        # Basic matcher scores
        features.extend([
            scores.get('general', 0.0),
            scores.get('skills', 0.0),
            scores.get('experience', 0.0),
            scores.get('location', 0.0),
            scores.get('education', 0.0)
        ])
        
        # Interaction features
        features.extend([
            scores.get('general', 0.0) * scores.get('skills', 0.0),  # General-Skills interaction
            scores.get('skills', 0.0) * scores.get('experience', 0.0),  # Skills-Experience interaction
            scores.get('education', 0.0) * scores.get('experience', 0.0),  # Education-Experience interaction
        ])
        
        # Statistical features
        score_values = [scores.get(dim, 0.0) for dim in ['general', 'skills', 'experience', 'location', 'education']]
        features.extend([
            np.mean(score_values),  # Mean score
            np.std(score_values),   # Score variance
            max(score_values),      # Max score
            min(score_values),      # Min score
            np.median(score_values) # Median score
        ])
        
        # Resume-specific features
        resume_features = self._extract_resume_features(resume)
        features.extend(resume_features)
        
        # Job-specific features
        job_features = self._extract_job_features(job)
        features.extend(job_features)
        
        # Content-based features
        content_features = self._extract_content_features(resume, job)
        features.extend(content_features)
        
        return np.array(features, dtype=float)

    def _extract_resume_features(self, resume: Dict[str, Any]) -> List[float]:
        """Extract features specific to the resume."""
        
        features = []
        
        # Basic resume statistics (with safe type conversion)
        char_len = self._safe_numeric_conversion(resume.get('char_len', 0))
        sent_len = self._safe_numeric_conversion(resume.get('sent_len', 0))
        type_token_ratio = self._safe_numeric_conversion(resume.get('type_token_ratio', 0.0))
        gender_term_count = self._safe_numeric_conversion(resume.get('gender_term_count', 0))
        
        features.append(char_len / 1000.0)  # Normalized character length
        features.append(sent_len / 50.0)    # Normalized sentence count
        features.append(type_token_ratio)   # Linguistic diversity
        features.append(gender_term_count / 10.0)  # Normalized gender terms
        
        # Category encoding (one-hot)
        category = resume.get('Category', 'UNKNOWN').upper()
        features.extend([
            1.0 if category == 'INFORMATION-TECHNOLOGY' else 0.0,
            1.0 if category == 'AUTOMOBILE' else 0.0,
            1.0 if category == 'HR' else 0.0
        ])
        
        # Parsed content features
        try:
            if isinstance(resume.get('parsed_json'), str):
                parsed_data = json.loads(resume['parsed_json'])
                
                # Education features
                education = parsed_data.get('education', [])
                features.extend([
                    len(education),  # Number of education entries
                    self._estimate_highest_degree_level(education),  # Highest degree level
                ])
                
                # Work experience features
                work = parsed_data.get('work', [])
                features.extend([
                    len(work),  # Number of work experiences
                    self._estimate_total_experience_years(work),  # Total years of experience
                ])
                
                # Skills features
                skills = parsed_data.get('skills', [])
                features.extend([
                    len(skills),  # Number of skills listed
                ])
                
            else:
                # Default values if parsing fails
                features.extend([0, 0, 0, 0, 0])
                
        except (json.JSONDecodeError, KeyError, TypeError):
            # Default values if parsing fails
            features.extend([0, 0, 0, 0, 0])
        
        return features

    def _extract_job_features(self, job: Dict[str, Any]) -> List[float]:
        """Extract features specific to the job."""
        
        features = []
        
        # Job description length
        desc_length = len(job.get('Long Description', ''))
        features.append(desc_length / 1000.0)  # Normalized description length
        
        # Experience requirement
        exp_years = job.get('Exp Years', '')
        exp_numeric = self._extract_numeric_experience(exp_years)
        features.append(exp_numeric / 10.0)  # Normalized experience requirement
        
        # Company name length (proxy for company size/establishment)
        company_length = len(job.get('Company Name', ''))
        features.append(company_length / 50.0)  # Normalized company name length
        
        # Primary keyword analysis
        primary_keyword = job.get('Primary Keyword', '').lower()
        features.extend([
            1.0 if 'senior' in primary_keyword else 0.0,
            1.0 if 'junior' in primary_keyword else 0.0,
            1.0 if 'manager' in primary_keyword else 0.0,
            1.0 if 'developer' in primary_keyword else 0.0,
            1.0 if 'engineer' in primary_keyword else 0.0
        ])
        
        return features

    def _extract_content_features(self, resume: Dict[str, Any], job: Dict[str, Any]) -> List[float]:
        """Extract features based on content similarity."""
        
        features = []
        
        # Text length ratio
        resume_text = resume.get('Resume_str', '')
        job_text = job.get('Long Description', '')
        
        resume_len = len(resume_text)
        job_len = len(job_text)
        
        if job_len > 0:
            features.append(resume_len / job_len)  # Length ratio
        else:
            features.append(0.0)
        
        # Simple keyword overlap
        resume_words = set(resume_text.lower().split())
        job_words = set(job_text.lower().split())
        
        if len(job_words) > 0:
            overlap = len(resume_words & job_words) / len(job_words)
            features.append(overlap)
        else:
            features.append(0.0)
        
        # Position title in resume check
        position = job.get('Position', '').lower()
        features.append(1.0 if position in resume_text.lower() else 0.0)
        
        return features

    def _estimate_highest_degree_level(self, education: List[Dict]) -> float:
        """Estimate highest degree level from education data."""
        
        degree_levels = {
            'phd': 5, 'doctorate': 5, 'doctoral': 5,
            'master': 4, 'masters': 4, 'mba': 4, 'ms': 4, 'ma': 4,
            'bachelor': 3, 'bachelors': 3, 'bs': 3, 'ba': 3,
            'associate': 2, 'associates': 2,
            'diploma': 1, 'certificate': 1, 'certification': 1
        }
        
        max_level = 0
        for edu in education:
            study_type = edu.get('studyType', '').lower()
            for degree_name, level in degree_levels.items():
                if degree_name in study_type:
                    max_level = max(max_level, level)
                    break
        
        return float(max_level)

    def _estimate_total_experience_years(self, work: List[Dict]) -> float:
        """Estimate total years of experience from work data."""
        
        total_years = 0
        current_year = 2024
        
        for work_item in work:
            start_date = work_item.get('startDate', '')
            end_date = work_item.get('endDate', '')
            
            start_year = self._extract_year(start_date)
            end_year = self._extract_year(end_date) if end_date else current_year
            
            if start_year and end_year:
                years = max(0, end_year - start_year)
                total_years += years
        
        return float(total_years)

    def _extract_year(self, date_string: str) -> Optional[int]:
        """Extract year from date string."""
        import re
        if not date_string:
            return None
        match = re.search(r'(\d{4})', str(date_string))
        return int(match.group(1)) if match else None

    def _extract_numeric_experience(self, exp_string: str) -> float:
        """Extract numeric experience requirement."""
        import re
        if not exp_string:
            return 0.0
        match = re.search(r'(\d+)', str(exp_string))
        return float(match.group(1)) if match else 0.0

    def _safe_numeric_conversion(self, value: Any) -> float:
        """Safely convert any value to float, handling strings and other types."""
        try:
            if value is None:
                return 0.0
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                # Try to extract numeric value from string
                import re
                if value.strip() == '' or value.lower() in ['none', 'null', 'n/a', 'na']:
                    return 0.0
                # Try direct conversion first
                try:
                    return float(value)
                except ValueError:
                    # Extract first number from string
                    match = re.search(r'(\d+\.?\d*)', value)
                    return float(match.group(1)) if match else 0.0
            return 0.0
        except (ValueError, AttributeError, TypeError):
            return 0.0

    def prepare_training_data(self, ranking_results: List[Dict], 
                            resumes_data: List[Dict], jobs_data: List[Dict],
                            feedback_data: Optional[List[Dict]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from ranking results and feedback.
        
        Args:
            ranking_results: List of ranking results
            resumes_data: Resume data
            jobs_data: Job data
            feedback_data: Optional feedback data for supervised learning
            
        Returns:
            Tuple of (features, labels) arrays
        """
        
        # Create lookup maps
        resume_map = {r['ID']: r for r in resumes_data}
        job_map = {j['id']: j for j in jobs_data}
        
        features_list = []
        labels_list = []
        
        # If we have feedback data, use it for supervised learning
        if feedback_data:
            feedback_map = {(f['resume_id'], f['job_id']): f['rating'] for f in feedback_data}
        
        for result in ranking_results:
            resume_id = result.get('resume_id')
            job_id = result.get('job_id')
            
            if resume_id not in resume_map or job_id not in job_map:
                continue
            
            resume = resume_map[resume_id]
            job = job_map[job_id]
            
            # Extract scores for feature computation
            scores = {
                'general': result.get('general_score', 0.0),
                'skills': result.get('skills_score', 0.0),
                'experience': result.get('experience_score', 0.0),
                'location': result.get('location_score', 0.0),
                'education': result.get('education_score', 0.0)
            }
            
            # Extract features
            features = self.extract_features(resume, job, scores)
            features_list.append(features)
            
            # Determine label
            if feedback_data and (resume_id, job_id) in feedback_map:
                # Use feedback rating as ground truth
                label = feedback_map[(resume_id, job_id)]
            else:
                # Use current total score as pseudo-label
                label = result.get('total_score', 0.0)
            
            labels_list.append(label)
        
        if not features_list:
            raise ValueError("No valid training data found")
        
        # Set feature names for interpretability
        self.feature_names = self._get_feature_names()
        
        return np.array(features_list), np.array(labels_list)

    def _get_feature_names(self) -> List[str]:
        """Get feature names for interpretability."""
        
        names = [
            # Basic scores
            'general_score', 'skills_score', 'experience_score', 'location_score', 'education_score',
            
            # Interaction features
            'general_skills_interaction', 'skills_experience_interaction', 'education_experience_interaction',
            
            # Statistical features
            'mean_score', 'score_std', 'max_score', 'min_score', 'median_score',
            
            # Resume features
            'resume_char_len_norm', 'resume_sent_len_norm', 'type_token_ratio', 'gender_term_count_norm',
            'category_IT', 'category_AUTO', 'category_HR',
            'num_education', 'highest_degree_level', 'num_work_exp', 'total_exp_years', 'num_skills',
            
            # Job features
            'job_desc_len_norm', 'exp_years_req_norm', 'company_name_len_norm',
            'is_senior', 'is_junior', 'is_manager', 'is_developer', 'is_engineer',
            
            # Content features
            'length_ratio', 'keyword_overlap', 'position_in_resume'
        ]
        
        return names

    def train_model(self, features: np.ndarray, labels: np.ndarray, 
                   validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the learning-to-rank model.
        
        Args:
            features: Feature matrix
            labels: Target labels/scores
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training results and metrics
        """
        
        try:
            # Initialize model
            if self.model_type not in self.model_options:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            self.model = self.model_options[self.model_type]
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                features, labels, test_size=validation_split, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_pred = self.model.predict(X_train_scaled)
            val_pred = self.model.predict(X_val_scaled)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, train_pred)
            val_mse = mean_squared_error(y_val, val_pred)
            
            # Feature importance (if available)
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            elif hasattr(self.model, 'coef_'):
                self.feature_importance = dict(zip(self.feature_names, abs(self.model.coef_)))
            
            # Cross-validation score
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
            
            results = {
                'train_mse': train_mse,
                'val_mse': val_mse,
                'cv_mean': -cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'feature_importance': self.feature_importance,
                'model_type': self.model_type,
                'training_samples': len(X_train),
                'validation_samples': len(X_val)
            }
            
            self.evaluation_metrics = results
            self.training_history.append(results)
            self.is_trained = True
            
            self.logger.info(f"Model trained successfully. Validation MSE: {val_mse:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            raise

    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        """
        Predict scores for given features.
        
        Args:
            features: Feature matrix
            
        Returns:
            Predicted scores
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)

    def rank_candidates(self, ranking_results: List[Dict], resumes_data: List[Dict], 
                       jobs_data: List[Dict]) -> List[Dict]:
        """
        Re-rank candidates using the trained model.
        
        Args:
            ranking_results: Original ranking results
            resumes_data: Resume data
            jobs_data: Job data
            
        Returns:
            Re-ranked results with ML scores
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before ranking")
        
        # Create lookup maps
        resume_map = {r['ID']: r for r in resumes_data}
        job_map = {j['id']: j for j in jobs_data}
        
        enhanced_results = []
        
        for result in ranking_results:
            resume_id = result.get('resume_id')
            job_id = result.get('job_id')
            
            if resume_id not in resume_map or job_id not in job_map:
                continue
            
            resume = resume_map[resume_id]
            job = job_map[job_id]
            
            # Extract scores
            scores = {
                'general': result.get('general_score', 0.0),
                'skills': result.get('skills_score', 0.0),
                'experience': result.get('experience_score', 0.0),
                'location': result.get('location_score', 0.0),
                'education': result.get('education_score', 0.0)
            }
            
            # Extract features and predict
            features = self.extract_features(resume, job, scores)
            ml_score = self.predict_scores(features.reshape(1, -1))[0]
            
            # Add ML score to result
            enhanced_result = result.copy()
            enhanced_result['ml_score'] = float(ml_score)
            enhanced_result['original_total_score'] = result.get('total_score', 0.0)
            
            enhanced_results.append(enhanced_result)
        
        # Re-rank by ML score
        enhanced_results.sort(key=lambda x: x['ml_score'], reverse=True)
        
        # Update ranks
        for i, result in enumerate(enhanced_results):
            result['ml_rank'] = i + 1
            result['rank_change'] = result.get('rank', 0) - (i + 1)
        
        return enhanced_results

    def get_feature_importance_report(self) -> str:
        """Generate a feature importance report."""
        
        if not self.feature_importance:
            return "No feature importance data available. Train the model first."
        
        # Sort features by importance
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        report = []
        report.append("=" * 50)
        report.append("FEATURE IMPORTANCE REPORT")
        report.append("=" * 50)
        report.append("")
        
        report.append("Top 10 Most Important Features:")
        report.append("-" * 35)
        
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            report.append(f"{i:2d}. {feature:<25} {importance:.4f}")
        
        report.append("")
        report.append("Feature Categories:")
        report.append("-" * 20)
        
        # Group by category
        categories = {
            'Basic Scores': ['general_score', 'skills_score', 'experience_score', 'location_score', 'education_score'],
            'Interactions': ['general_skills_interaction', 'skills_experience_interaction', 'education_experience_interaction'],
            'Statistical': ['mean_score', 'score_std', 'max_score', 'min_score', 'median_score'],
            'Resume Features': [f for f in self.feature_names if f.startswith(('resume_', 'category_', 'num_', 'highest_', 'total_'))],
            'Job Features': [f for f in self.feature_names if f.startswith(('job_', 'exp_years', 'company_', 'is_'))],
            'Content Features': ['length_ratio', 'keyword_overlap', 'position_in_resume']
        }
        
        for category, features in categories.items():
            category_importance = sum(self.feature_importance.get(f, 0) for f in features)
            report.append(f"{category:<20} {category_importance:.4f}")
        
        return "\n".join(report)

    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'evaluation_metrics': self.evaluation_metrics,
            'training_history': self.training_history
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.evaluation_metrics = model_data['evaluation_metrics']
        self.training_history = model_data['training_history']
        self.is_trained = True
        
        self.logger.info(f"Model loaded from {filepath}")

    def evaluate_ranking_quality(self, original_results: List[Dict], 
                                ml_results: List[Dict], 
                                feedback_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Evaluate the quality of ML-based ranking vs original ranking.
        
        Args:
            original_results: Original ranking results
            ml_results: ML re-ranked results
            feedback_data: Optional ground truth feedback
            
        Returns:
            Evaluation metrics
        """
        
        evaluation = {
            'ranking_correlation': self._calculate_ranking_correlation(original_results, ml_results),
            'score_improvement': self._calculate_score_improvement(original_results, ml_results),
            'rank_changes': self._analyze_rank_changes(ml_results)
        }
        
        if feedback_data:
            evaluation['ground_truth_evaluation'] = self._evaluate_against_ground_truth(
                ml_results, feedback_data
            )
        
        return evaluation

    def _calculate_ranking_correlation(self, original: List[Dict], ml_ranked: List[Dict]) -> float:
        """Calculate correlation between original and ML rankings."""
        
        from scipy.stats import spearmanr
        
        # Create mapping of resume_id to ranks
        original_ranks = {r['resume_id']: r.get('rank', 0) for r in original}
        ml_ranks = {r['resume_id']: r.get('ml_rank', 0) for r in ml_ranked}
        
        common_ids = set(original_ranks.keys()) & set(ml_ranks.keys())
        
        if len(common_ids) < 2:
            return 0.0
        
        orig_rank_list = [original_ranks[rid] for rid in common_ids]
        ml_rank_list = [ml_ranks[rid] for rid in common_ids]
        
        correlation, _ = spearmanr(orig_rank_list, ml_rank_list)
        return correlation if not np.isnan(correlation) else 0.0

    def _calculate_score_improvement(self, original: List[Dict], ml_ranked: List[Dict]) -> Dict[str, float]:
        """Calculate score improvements from ML ranking."""
        
        original_scores = [r.get('total_score', 0) for r in original]
        ml_scores = [r.get('ml_score', 0) for r in ml_ranked]
        
        return {
            'original_mean': np.mean(original_scores),
            'ml_mean': np.mean(ml_scores),
            'improvement': np.mean(ml_scores) - np.mean(original_scores),
            'original_std': np.std(original_scores),
            'ml_std': np.std(ml_scores)
        }

    def _analyze_rank_changes(self, ml_results: List[Dict]) -> Dict[str, Any]:
        """Analyze rank changes from ML re-ranking."""
        
        rank_changes = [r.get('rank_change', 0) for r in ml_results]
        
        return {
            'mean_rank_change': np.mean(rank_changes),
            'max_improvement': max(rank_changes),
            'max_decline': min(rank_changes),
            'candidates_improved': sum(1 for change in rank_changes if change > 0),
            'candidates_declined': sum(1 for change in rank_changes if change < 0),
            'candidates_unchanged': sum(1 for change in rank_changes if change == 0)
        }

    def _evaluate_against_ground_truth(self, ml_results: List[Dict], 
                                     feedback_data: List[Dict]) -> Dict[str, float]:
        """Evaluate ML rankings against ground truth feedback."""
        
        # Create feedback mapping
        feedback_map = {(f['resume_id'], f['job_id']): f['rating'] for f in feedback_data}
        
        # Extract relevant results and ground truth
        ml_scores = []
        ground_truth = []
        
        for result in ml_results:
            resume_id = result.get('resume_id')
            job_id = result.get('job_id')
            
            if (resume_id, job_id) in feedback_map:
                ml_scores.append(result.get('ml_score', 0))
                ground_truth.append(feedback_map[(resume_id, job_id)])
        
        if len(ml_scores) < 2:
            return {'error': 'Insufficient ground truth data'}
        
        # Calculate metrics
        mse = mean_squared_error(ground_truth, ml_scores)
        correlation = np.corrcoef(ground_truth, ml_scores)[0, 1] if len(ml_scores) > 1 else 0.0
        
        return {
            'mse_vs_ground_truth': mse,
            'correlation_vs_ground_truth': correlation if not np.isnan(correlation) else 0.0,
            'samples_evaluated': len(ml_scores)
        } 