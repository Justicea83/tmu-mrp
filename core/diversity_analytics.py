"""
Diversity Analytics Module for Resume-Job Ranking System

This module analyzes diversity and bias metrics in ranking results to ensure
fair and equitable candidate evaluation.
"""

import json
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter, defaultdict
import logging


class DiversityAnalytics:
    """
    Analyzes diversity metrics and potential bias in ranking results.
    
    Features:
    - Gender representation analysis
    - Category bias detection
    - Score distribution fairness
    - Statistical significance testing
    - Bias mitigation recommendations
    - Gender-coded language analysis (Gaucher et al. 2011)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Gender detection patterns (basic heuristics)
        self.gender_indicators = {
            'female': ['she', 'her', 'hers', 'woman', 'female', 'girl', 'lady', 'ms.', 'mrs.'],
            'male': ['he', 'him', 'his', 'man', 'male', 'boy', 'gentleman', 'mr.', 'sir']
        }
        
        # Name-based gender indicators (simplified - in production would use more comprehensive database)
        self.female_names = {
            'jennifer', 'lisa', 'susan', 'karen', 'nancy', 'donna', 'carol', 'sandra', 'ruth',
            'sarah', 'jessica', 'elizabeth', 'tina', 'emily', 'amanda', 'maria', 'michelle',
            'stephanie', 'angela', 'brenda', 'emma', 'olivia', 'ava', 'sophia', 'isabella'
        }
        
        self.male_names = {
            'james', 'john', 'robert', 'michael', 'william', 'david', 'richard', 'charles',
            'joseph', 'thomas', 'christopher', 'daniel', 'paul', 'mark', 'donald', 'steven',
            'andrew', 'joshua', 'kenneth', 'kevin', 'brian', 'george', 'edward', 'ronald'
        }
        
        # Gaucher et al. (2011) Gender-coded word lists
        # From "Evidence That Gendered Wording in Job Advertisements Exists and Sustains Gender Inequality"
        self.masculine_words = [
            "active", "adventurous", "aggress", "ambitio", "analy", "assert", "athlet", "autonom", 
            "battle", "boast", "challeng", "compet", "confident", "courag", "decid", "decision", 
            "decisive", "defend", "determin", "dominant", "force", "greedy", "headstrong", "hierarch", 
            "hostil", "impulsive", "independen", "individual", "intellect", "lead", "logic", 
            "objective", "opinion", "outspoken", "persist", "principle", "reckless", "self", 
            "self-reliant", "stubborn", "superior"
        ]
        
        self.feminine_words = [
            "affectionate", "child", "cheer", "commit", "communal", "compassion", "connect", 
            "considerate", "cooperat", "depend", "emotion", "empath", "feel", "flatterable", 
            "gentle", "honest", "interdependen", "interpersona", "kind", "kinship", "loyal", 
            "modesty", "nag", "nurtur", "pleasant", "polite", "quiet", "respon", "sensitiv", 
            "submissive", "support", "sympath", "tender", "together", "trust", "understand", 
            "warm", "whin", "yield"
        ]
        
        # Statistical significance thresholds
        self.significance_threshold = 0.05
        self.effect_size_thresholds = {
            'small': 0.2,
            'medium': 0.5,
            'large': 0.8
        }

    def analyze_diversity_metrics(self, results: List[Dict[str, Any]], 
                                 resumes_data: List[Dict[str, Any]],
                                 jobs_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Comprehensive diversity analysis of ranking results.
        
        Args:
            results: List of ranking results
            resumes_data: List of resume data for additional analysis
            jobs_data: List of job data for gender-coded language analysis
            
        Returns:
            Dict containing comprehensive diversity metrics
        """
        
        analysis = {
            'summary': self._generate_diversity_summary(results, resumes_data),
            'gender_analysis': self._analyze_gender_representation(results, resumes_data),
            'category_analysis': self._analyze_category_bias(results),
            'score_distribution': self._analyze_score_distributions(results),
            'ranking_bias': self._analyze_ranking_bias(results),
            'statistical_tests': self._perform_statistical_tests(results, resumes_data),
            'recommendations': self._generate_bias_recommendations(results, resumes_data)
        }
        
        # Add gender-coded language analysis if job data is provided
        if jobs_data:
            analysis['gender_coded_language'] = self._analyze_gender_coded_language(jobs_data)
        
        return analysis

    def _generate_diversity_summary(self, results: List[Dict], 
                                   resumes_data: List[Dict]) -> Dict[str, Any]:
        """Generate high-level diversity summary."""
        
        # Category distribution
        categories = [r.get('resume_category', 'Unknown') for r in results]
        category_dist = Counter(categories)
        
        # Gender estimation
        gender_estimates = []
        for resume in resumes_data:
            gender = self._estimate_gender(resume)
            gender_estimates.append(gender)
        
        gender_dist = Counter(gender_estimates)
        
        # Calculate diversity scores
        category_diversity = self._calculate_diversity_index(list(category_dist.values()))
        gender_diversity = self._calculate_diversity_index(list(gender_dist.values()))
        
        return {
            'total_candidates': len(results),
            'unique_categories': len(category_dist),
            'category_distribution': dict(category_dist),
            'category_diversity_index': round(category_diversity, 3),
            'estimated_gender_distribution': dict(gender_dist),
            'gender_diversity_index': round(gender_diversity, 3),
            'diversity_assessment': self._assess_overall_diversity(category_diversity, gender_diversity)
        }

    def _estimate_gender(self, resume: Dict[str, Any]) -> str:
        """Estimate gender from resume content (basic heuristics)."""
        
        try:
            # Check parsed JSON for name
            if isinstance(resume.get('parsed_json'), str):
                parsed_data = json.loads(resume['parsed_json'])
                name = parsed_data.get('basics', {}).get('name')
                
                if name and isinstance(name, str):
                    name_lower = name.lower()
                    first_name = name_lower.split()[0] if name_lower.split() else ''
                    if first_name in self.female_names:
                        return 'Female'
                    elif first_name in self.male_names:
                        return 'Male'
            
            # Check resume text for gender indicators
            resume_str = resume.get('Resume_str', '') or ''
            text_from_html = resume.get('text_from_html', '') or ''
            resume_text = (resume_str + ' ' + text_from_html).lower()
            
            female_count = sum(1 for indicator in self.gender_indicators['female'] 
                             if indicator in resume_text)
            male_count = sum(1 for indicator in self.gender_indicators['male'] 
                           if indicator in resume_text)
            
            if female_count > male_count and female_count > 0:
                return 'Female'
            elif male_count > female_count and male_count > 0:
                return 'Male'
            
            return 'Unknown'
            
        except Exception as e:
            self.logger.warning(f"Error estimating gender: {e}")
            return 'Unknown'
    
    def _analyze_gender_coded_language(self, jobs_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze gender-coded language in job descriptions using Gaucher et al. (2011) methodology.
        
        Args:
            jobs_data: List of job posting data
            
        Returns:
            Dict containing gender bias analysis for each job
        """
        
        job_analyses = []
        overall_stats = {
            'masculine_bias_count': 0,
            'feminine_bias_count': 0,
            'neutral_count': 0,
            'total_jobs': len(jobs_data)
        }
        
        for job in jobs_data:
            job_analysis = self._analyze_single_job_gender_coding(job)
            job_analyses.append(job_analysis)
            
            # Update overall statistics
            if job_analysis['gender_polarity'] > 1:
                overall_stats['masculine_bias_count'] += 1
            elif job_analysis['gender_polarity'] < -1:
                overall_stats['feminine_bias_count'] += 1
            else:
                overall_stats['neutral_count'] += 1
        
        # Calculate percentages
        total = overall_stats['total_jobs']
        if total > 0:
            overall_stats['masculine_bias_percentage'] = round((overall_stats['masculine_bias_count'] / total) * 100, 1)
            overall_stats['feminine_bias_percentage'] = round((overall_stats['feminine_bias_count'] / total) * 100, 1)
            overall_stats['neutral_percentage'] = round((overall_stats['neutral_count'] / total) * 100, 1)
        
        return {
            'methodology': 'Gaucher et al. (2011) - Evidence That Gendered Wording in Job Advertisements Exists and Sustains Gender Inequality',
            'overall_statistics': overall_stats,
            'job_analyses': job_analyses,
            'bias_assessment': self._assess_gender_bias_severity(overall_stats),
            'recommendations': self._generate_gender_language_recommendations(overall_stats, job_analyses)
        }
    
    def _analyze_single_job_gender_coding(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze gender-coded language for a single job posting.
        
        Args:
            job: Single job posting data
            
        Returns:
            Dict containing gender bias metrics for the job
        """
        
        # Extract job text
        job_text = self._extract_job_text(job).lower()
        
        # Count masculine and feminine words
        masculine_score = sum(1 for word in self.masculine_words if word in job_text)
        feminine_score = sum(1 for word in self.feminine_words if word in job_text)
        
        # Calculate gender polarity (Gaucher et al. methodology)
        gender_polarity = masculine_score - feminine_score
        
        # Identify specific words found
        masculine_words_found = [word for word in self.masculine_words if word in job_text]
        feminine_words_found = [word for word in self.feminine_words if word in job_text]
        
        # Determine bias classification
        bias_classification = self._classify_gender_bias(gender_polarity)
        
        return {
            'job_id': job.get('id', 'Unknown'),
            'job_title': job.get('Position', 'Unknown'),
            'company': job.get('Company', 'Unknown'),
            'masculine_score': masculine_score,
            'feminine_score': feminine_score,
            'gender_polarity': gender_polarity,
            'bias_classification': bias_classification,
            'masculine_words_found': masculine_words_found,
            'feminine_words_found': feminine_words_found,
            'total_gendered_words': masculine_score + feminine_score,
            'word_density': self._calculate_word_density(job_text, masculine_score + feminine_score)
        }
    
    def _extract_job_text(self, job: Dict[str, Any]) -> str:
        """Extract all relevant text from job posting."""
        
        text_fields = [
            'Position', 'Company', 'Long Description', 'Short Description',
            'Primary Keyword', 'Requirements', 'Responsibilities', 'Benefits'
        ]
        
        job_text = ""
        for field in text_fields:
            value = job.get(field)
            if value and isinstance(value, str):
                job_text += " " + value
        
        return job_text.strip()
    
    def _classify_gender_bias(self, gender_polarity: int) -> str:
        """
        Classify gender bias level based on polarity score.
        
        Args:
            gender_polarity: Masculine score minus feminine score
            
        Returns:
            String classification of bias level
        """
        
        if gender_polarity >= 3:
            return "Strong Masculine Bias"
        elif gender_polarity >= 1:
            return "Moderate Masculine Bias"
        elif gender_polarity <= -3:
            return "Strong Feminine Bias"
        elif gender_polarity <= -1:
            return "Moderate Feminine Bias"
        else:
            return "Gender Neutral"
    
    def _calculate_word_density(self, text: str, gendered_word_count: int) -> float:
        """Calculate density of gendered words per 100 words."""
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        return round((gendered_word_count / total_words) * 100, 2)
    
    def _assess_gender_bias_severity(self, stats: Dict[str, Any]) -> str:
        """Assess overall gender bias severity across all jobs."""
        
        total = stats['total_jobs']
        if total == 0:
            return "No data available"
        
        masculine_pct = stats.get('masculine_bias_percentage', 0)
        feminine_pct = stats.get('feminine_bias_percentage', 0)
        
        if masculine_pct > 50 or feminine_pct > 50:
            return "High Bias Risk - Majority of jobs show gender-coded language"
        elif masculine_pct > 25 or feminine_pct > 25:
            return "Moderate Bias Risk - Significant portion of jobs show gender-coded language"
        elif masculine_pct > 10 or feminine_pct > 10:
            return "Low Bias Risk - Some jobs show gender-coded language"
        else:
            return "Minimal Bias Risk - Most jobs use gender-neutral language"
    
    def _generate_gender_language_recommendations(self, stats: Dict[str, Any], 
                                                job_analyses: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for reducing gender-coded language."""
        
        recommendations = []
        
        # Overall recommendations
        masculine_pct = stats.get('masculine_bias_percentage', 0)
        feminine_pct = stats.get('feminine_bias_percentage', 0)
        
        if masculine_pct > 25:
            recommendations.append(
                f"CRITICAL: {masculine_pct}% of jobs show masculine bias. "
                "Replace competitive/aggressive language with collaborative terms."
            )
        
        if feminine_pct > 25:
            recommendations.append(
                f"ATTENTION: {feminine_pct}% of jobs show feminine bias. "
                "Balance nurturing language with achievement-oriented terms."
            )
        
        # Specific word recommendations
        most_common_masculine = self._find_most_common_words(job_analyses, 'masculine_words_found')
        most_common_feminine = self._find_most_common_words(job_analyses, 'feminine_words_found')
        
        if most_common_masculine:
            recommendations.append(
                f"Most common masculine words: {', '.join(most_common_masculine[:5])}. "
                "Consider alternatives: 'lead' → 'guide', 'dominant' → 'effective', 'aggressive' → 'proactive'"
            )
        
        if most_common_feminine:
            recommendations.append(
                f"Most common feminine words: {', '.join(most_common_feminine[:5])}. "
                "Balance with achievement terms: Add 'achieve', 'accomplish', 'drive results'"
            )
        
        # General best practices
        recommendations.extend([
            "Use active voice and specific action verbs",
            "Focus on role requirements rather than personal traits",
            "Include both collaborative and achievement-oriented language",
            "Review job postings with diverse team members before publishing"
        ])
        
        return recommendations
    
    def _find_most_common_words(self, job_analyses: List[Dict[str, Any]], 
                               word_field: str) -> List[str]:
        """Find most commonly used gendered words across all jobs."""
        
        word_counts = Counter()
        for analysis in job_analyses:
            words = analysis.get(word_field, [])
            word_counts.update(words)
        
        return [word for word, count in word_counts.most_common(10)]
    
    def gender_polarity_counts(self, text: str) -> Dict[str, int]:
        """
        Gaucher et al. (2011) gender polarity analysis function.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with masculine_score, feminine_score, and gender_polarity
        """
        
        text = text.lower()
        masculine_score = sum(text.count(word) for word in self.masculine_words)
        feminine_score = sum(text.count(word) for word in self.feminine_words)
        
        return {
            "masculine_score": masculine_score,
            "feminine_score": feminine_score,
            "gender_polarity": masculine_score - feminine_score
        }

    def _calculate_diversity_index(self, counts: List[int]) -> float:
        """Calculate Shannon diversity index."""
        
        if not counts or sum(counts) == 0:
            return 0.0
        
        total = sum(counts)
        proportions = [count / total for count in counts if count > 0]
        
        if len(proportions) <= 1:
            return 0.0
        
        # Shannon diversity index
        diversity = -sum(p * np.log(p) for p in proportions)
        
        # Normalize by maximum possible diversity
        max_diversity = np.log(len(proportions))
        
        return diversity / max_diversity if max_diversity > 0 else 0.0

    def _assess_overall_diversity(self, category_diversity: float, 
                                 gender_diversity: float) -> str:
        """Assess overall diversity level."""
        
        avg_diversity = (category_diversity + gender_diversity) / 2
        
        if avg_diversity >= 0.8:
            return 'High Diversity'
        elif avg_diversity >= 0.6:
            return 'Moderate Diversity'
        elif avg_diversity >= 0.4:
            return 'Low Diversity'
        else:
            return 'Very Low Diversity'

    def _analyze_gender_representation(self, results: List[Dict], 
                                     resumes_data: List[Dict]) -> Dict[str, Any]:
        """Analyze gender representation in results."""
        
        # Create mapping of resume_id to gender
        resume_gender_map = {}
        for resume in resumes_data:
            resume_id = resume.get('ID')
            gender = self._estimate_gender(resume)
            resume_gender_map[resume_id] = gender
        
        # Analyze gender in top rankings
        gender_by_rank = defaultdict(list)
        gender_scores = defaultdict(list)
        
        for result in results:
            resume_id = result.get('resume_id')
            rank = result.get('rank', 0)
            total_score = result.get('total_score', 0)
            
            gender = resume_gender_map.get(resume_id, 'Unknown')
            gender_by_rank[rank].append(gender)
            gender_scores[gender].append(total_score)
        
        # Calculate representation metrics
        top_5_gender = []
        top_10_gender = []
        
        for result in results:
            resume_id = result.get('resume_id')
            rank = result.get('rank', 0)
            gender = resume_gender_map.get(resume_id, 'Unknown')
            
            if rank <= 5:
                top_5_gender.append(gender)
            if rank <= 10:
                top_10_gender.append(gender)
        
        analysis = {
            'overall_gender_distribution': dict(Counter(resume_gender_map.values())),
            'top_5_gender_distribution': dict(Counter(top_5_gender)),
            'top_10_gender_distribution': dict(Counter(top_10_gender)),
            'average_scores_by_gender': {
                gender: round(np.mean(scores), 3) if scores else 0
                for gender, scores in gender_scores.items()
            },
            'gender_score_variance': {
                gender: round(np.var(scores), 4) if scores else 0
                for gender, scores in gender_scores.items()
            }
        }
        
        # Check for significant differences
        if len(gender_scores.get('Female', [])) > 0 and len(gender_scores.get('Male', [])) > 0:
            analysis['statistical_comparison'] = self._compare_gender_scores(
                gender_scores['Female'], gender_scores['Male']
            )
        
        return analysis

    def _compare_gender_scores(self, female_scores: List[float], 
                              male_scores: List[float]) -> Dict[str, Any]:
        """Compare scores between genders using statistical tests."""
        
        try:
            from scipy import stats
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(female_scores, male_scores)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(female_scores) - 1) * np.var(female_scores) + 
                                 (len(male_scores) - 1) * np.var(male_scores)) / 
                                (len(female_scores) + len(male_scores) - 2))
            
            effect_size = abs(np.mean(female_scores) - np.mean(male_scores)) / pooled_std if pooled_std > 0 else 0
            
            # Interpret results
            is_significant = p_value < self.significance_threshold
            effect_magnitude = 'large' if effect_size >= self.effect_size_thresholds['large'] else \
                              'medium' if effect_size >= self.effect_size_thresholds['medium'] else \
                              'small' if effect_size >= self.effect_size_thresholds['small'] else 'negligible'
            
            return {
                't_statistic': round(t_stat, 4),
                'p_value': round(p_value, 4),
                'effect_size': round(effect_size, 4),
                'effect_magnitude': effect_magnitude,
                'is_significant': is_significant,
                'interpretation': self._interpret_gender_comparison(is_significant, effect_magnitude, 
                                                                  np.mean(female_scores), np.mean(male_scores))
            }
            
        except ImportError:
            return {'error': 'scipy not available for statistical tests'}
        except Exception as e:
            self.logger.warning(f"Error in gender score comparison: {e}")
            return {'error': 'Could not perform statistical comparison'}

    def _interpret_gender_comparison(self, is_significant: bool, effect_magnitude: str,
                                   female_mean: float, male_mean: float) -> str:
        """Interpret gender comparison results."""
        
        if not is_significant:
            return "No statistically significant difference in scores between genders"
        
        higher_gender = "Female" if female_mean > male_mean else "Male"
        lower_gender = "Male" if female_mean > male_mean else "Female"
        
        if effect_magnitude in ['large', 'medium']:
            return f"Significant bias detected: {higher_gender} candidates score {effect_magnitude}ly higher than {lower_gender} candidates"
        else:
            return f"Statistically significant but small difference: {higher_gender} candidates score slightly higher"

    def _analyze_category_bias(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze bias across resume categories."""
        
        category_scores = defaultdict(list)
        category_ranks = defaultdict(list)
        
        for result in results:
            category = result.get('resume_category', 'Unknown')
            total_score = result.get('total_score', 0)
            rank = result.get('rank', 0)
            
            category_scores[category].append(total_score)
            category_ranks[category].append(rank)
        
        analysis = {
            'category_performance': {},
            'ranking_distribution': {},
            'bias_indicators': {}
        }
        
        # Calculate performance metrics by category
        for category, scores in category_scores.items():
            if scores:
                analysis['category_performance'][category] = {
                    'count': len(scores),
                    'mean_score': round(np.mean(scores), 3),
                    'median_score': round(np.median(scores), 3),
                    'std_score': round(np.std(scores), 3),
                    'min_score': round(min(scores), 3),
                    'max_score': round(max(scores), 3)
                }
        
        # Calculate ranking distribution
        for category, ranks in category_ranks.items():
            if ranks:
                analysis['ranking_distribution'][category] = {
                    'mean_rank': round(np.mean(ranks), 2),
                    'median_rank': round(np.median(ranks), 2),
                    'top_5_count': sum(1 for rank in ranks if rank <= 5),
                    'top_10_count': sum(1 for rank in ranks if rank <= 10),
                    'bottom_10_percent': sum(1 for rank in ranks if rank > len(results) * 0.9)
                }
        
        # Identify bias indicators
        analysis['bias_indicators'] = self._identify_category_bias_indicators(
            analysis['category_performance'], analysis['ranking_distribution']
        )
        
        return analysis

    def _identify_category_bias_indicators(self, performance: Dict, ranking: Dict) -> Dict[str, Any]:
        """Identify potential bias indicators across categories."""
        
        indicators = {
            'potential_bias': False,
            'concerns': [],
            'advantages': [],
            'recommendations': []
        }
        
        if len(performance) < 2:
            return indicators
        
        # Compare performance across categories
        scores_by_category = {cat: data['mean_score'] for cat, data in performance.items()}
        max_score = max(scores_by_category.values())
        min_score = min(scores_by_category.values())
        
        # Check for large disparities
        if max_score - min_score > 0.3:  # Threshold for concerning disparity
            indicators['potential_bias'] = True
            
            best_category = max(scores_by_category, key=scores_by_category.get)
            worst_category = min(scores_by_category, key=scores_by_category.get)
            
            indicators['concerns'].append(
                f"Large score disparity: {best_category} ({max_score:.3f}) vs {worst_category} ({min_score:.3f})"
            )
        
        # Check ranking distribution
        for category, rank_data in ranking.items():
            total_candidates = sum(perf['count'] for perf in performance.values())
            category_count = performance[category]['count']
            expected_top_5 = (category_count / total_candidates) * 5
            actual_top_5 = rank_data['top_5_count']
            
            if actual_top_5 > expected_top_5 * 1.5:  # Overrepresented
                indicators['advantages'].append(
                    f"{category} overrepresented in top 5 (expected {expected_top_5:.1f}, actual {actual_top_5})"
                )
            elif actual_top_5 < expected_top_5 * 0.5:  # Underrepresented
                indicators['concerns'].append(
                    f"{category} underrepresented in top 5 (expected {expected_top_5:.1f}, actual {actual_top_5})"
                )
        
        # Generate recommendations
        if indicators['potential_bias'] or indicators['concerns']:
            indicators['recommendations'].extend([
                "Review matching algorithms for category-specific bias",
                "Consider category-balanced sampling in training data",
                "Implement bias-aware ranking adjustments",
                "Conduct deeper statistical analysis of category differences"
            ])
        
        return indicators

    def _analyze_score_distributions(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze score distributions for fairness indicators."""
        
        # Extract scores by dimension
        dimension_scores = {
            'general': [r.get('general_score', 0) for r in results],
            'skills': [r.get('skills_score', 0) for r in results],
            'experience': [r.get('experience_score', 0) for r in results],
            'location': [r.get('location_score', 0) for r in results],
            'total': [r.get('total_score', 0) for r in results]
        }
        
        # Add education scores if available
        if any('education_score' in r for r in results):
            dimension_scores['education'] = [r.get('education_score', 0) for r in results]
        
        analysis = {}
        
        for dimension, scores in dimension_scores.items():
            if scores:
                analysis[dimension] = {
                    'distribution_stats': {
                        'mean': round(np.mean(scores), 3),
                        'median': round(np.median(scores), 3),
                        'std': round(np.std(scores), 3),
                        'skewness': round(self._calculate_skewness(scores), 3),
                        'kurtosis': round(self._calculate_kurtosis(scores), 3)
                    },
                    'fairness_indicators': {
                        'concentration_top_10pct': self._calculate_concentration(scores, 0.1),
                        'concentration_top_25pct': self._calculate_concentration(scores, 0.25),
                        'gini_coefficient': self._calculate_gini_coefficient(scores)
                    }
                }
        
        return analysis

    def _calculate_skewness(self, scores: List[float]) -> float:
        """Calculate skewness of score distribution."""
        mean = np.mean(scores)
        std = np.std(scores)
        if std == 0:
            return 0
        return np.mean([(x - mean) ** 3 for x in scores]) / (std ** 3)

    def _calculate_kurtosis(self, scores: List[float]) -> float:
        """Calculate kurtosis of score distribution."""
        mean = np.mean(scores)
        std = np.std(scores)
        if std == 0:
            return 0
        return np.mean([(x - mean) ** 4 for x in scores]) / (std ** 4) - 3

    def _calculate_concentration(self, scores: List[float], percentile: float) -> float:
        """Calculate what percentage of total score is held by top percentile."""
        sorted_scores = sorted(scores, reverse=True)
        top_n = int(len(scores) * percentile)
        if top_n == 0:
            return 0.0
        
        top_sum = sum(sorted_scores[:top_n])
        total_sum = sum(scores)
        
        return (top_sum / total_sum) if total_sum > 0 else 0.0

    def _calculate_gini_coefficient(self, scores: List[float]) -> float:
        """Calculate Gini coefficient for score inequality."""
        sorted_scores = sorted(scores)
        n = len(scores)
        if n == 0:
            return 0.0
        
        cum_scores = np.cumsum(sorted_scores)
        total_score = cum_scores[-1]
        
        if total_score == 0:
            return 0.0
        
        gini = (2 * sum((i + 1) * score for i, score in enumerate(sorted_scores))) / (n * total_score) - (n + 1) / n
        return max(0.0, gini)

    def _analyze_ranking_bias(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze potential bias in ranking positions."""
        
        # Group by resume category and analyze rank distributions
        category_ranks = defaultdict(list)
        for result in results:
            category = result.get('resume_category', 'Unknown')
            rank = result.get('rank', 0)
            category_ranks[category].append(rank)
        
        analysis = {
            'rank_distribution_by_category': {},
            'mobility_analysis': {},
            'bias_score': 0.0
        }
        
        # Calculate rank distribution statistics
        for category, ranks in category_ranks.items():
            if ranks:
                analysis['rank_distribution_by_category'][category] = {
                    'mean_rank': round(np.mean(ranks), 2),
                    'median_rank': round(np.median(ranks), 2),
                    'rank_variance': round(np.var(ranks), 2),
                    'best_rank': min(ranks),
                    'worst_rank': max(ranks),
                    'count': len(ranks)
                }
        
        # Calculate overall bias score
        if len(category_ranks) > 1:
            mean_ranks = [np.mean(ranks) for ranks in category_ranks.values()]
            rank_variance = np.var(mean_ranks)
            analysis['bias_score'] = round(rank_variance, 3)
        
        return analysis

    def _perform_statistical_tests(self, results: List[Dict], 
                                  resumes_data: List[Dict]) -> Dict[str, Any]:
        """Perform statistical tests for bias detection."""
        
        tests = {}
        
        try:
            # Test for category bias using ANOVA
            category_scores = defaultdict(list)
            for result in results:
                category = result.get('resume_category', 'Unknown')
                total_score = result.get('total_score', 0)
                category_scores[category].append(total_score)
            
            if len(category_scores) > 2:
                tests['category_anova'] = self._perform_anova_test(category_scores)
            
            # Test for gender bias
            resume_gender_map = {resume.get('ID'): self._estimate_gender(resume) 
                               for resume in resumes_data}
            
            gender_scores = defaultdict(list)
            for result in results:
                resume_id = result.get('resume_id')
                gender = resume_gender_map.get(resume_id, 'Unknown')
                if gender in ['Male', 'Female']:
                    total_score = result.get('total_score', 0)
                    gender_scores[gender].append(total_score)
            
            if len(gender_scores) == 2 and all(len(scores) > 5 for scores in gender_scores.values()):
                tests['gender_ttest'] = self._perform_gender_ttest(gender_scores)
            
        except Exception as e:
            self.logger.warning(f"Error performing statistical tests: {e}")
            tests['error'] = str(e)
        
        return tests

    def _perform_anova_test(self, category_scores: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform ANOVA test for category differences."""
        
        try:
            from scipy import stats
            
            score_groups = [scores for scores in category_scores.values() if len(scores) > 1]
            
            if len(score_groups) < 2:
                return {'error': 'Insufficient data for ANOVA test'}
            
            f_stat, p_value = stats.f_oneway(*score_groups)
            
            return {
                'f_statistic': round(f_stat, 4),
                'p_value': round(p_value, 4),
                'is_significant': p_value < self.significance_threshold,
                'interpretation': 'Significant category differences detected' if p_value < self.significance_threshold 
                               else 'No significant category differences detected'
            }
            
        except ImportError:
            return {'error': 'scipy not available for ANOVA test'}
        except Exception as e:
            return {'error': f'ANOVA test failed: {str(e)}'}

    def _perform_gender_ttest(self, gender_scores: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform t-test for gender differences."""
        
        try:
            from scipy import stats
            
            male_scores = gender_scores.get('Male', [])
            female_scores = gender_scores.get('Female', [])
            
            if len(male_scores) < 2 or len(female_scores) < 2:
                return {'error': 'Insufficient data for t-test'}
            
            t_stat, p_value = stats.ttest_ind(male_scores, female_scores)
            
            return {
                't_statistic': round(t_stat, 4),
                'p_value': round(p_value, 4),
                'is_significant': p_value < self.significance_threshold,
                'male_mean': round(np.mean(male_scores), 3),
                'female_mean': round(np.mean(female_scores), 3),
                'interpretation': 'Significant gender differences detected' if p_value < self.significance_threshold 
                               else 'No significant gender differences detected'
            }
            
        except ImportError:
            return {'error': 'scipy not available for t-test'}
        except Exception as e:
            return {'error': f't-test failed: {str(e)}'}

    def _generate_bias_recommendations(self, results: List[Dict], 
                                     resumes_data: List[Dict]) -> Dict[str, Any]:
        """Generate recommendations for addressing bias."""
        
        recommendations = {
            'immediate_actions': [],
            'algorithm_improvements': [],
            'data_collection': [],
            'monitoring': []
        }
        
        # Analyze current state
        category_dist = Counter(r.get('resume_category', 'Unknown') for r in results)
        diversity_index = self._calculate_diversity_index(list(category_dist.values()))
        
        # Generate specific recommendations
        if diversity_index < 0.6:
            recommendations['immediate_actions'].append(
                "Implement balanced sampling to ensure fair representation across categories"
            )
        
        # Check for score disparities
        category_scores = defaultdict(list)
        for result in results:
            category = result.get('resume_category', 'Unknown')
            total_score = result.get('total_score', 0)
            category_scores[category].append(total_score)
        
        if len(category_scores) > 1:
            score_means = [np.mean(scores) for scores in category_scores.values()]
            if max(score_means) - min(score_means) > 0.3:
                recommendations['algorithm_improvements'].extend([
                    "Review matching algorithms for category-specific bias",
                    "Consider separate model training for different resume categories",
                    "Implement bias-aware post-processing adjustments"
                ])
        
        # General recommendations
        recommendations['data_collection'].extend([
            "Collect more diverse training data",
            "Implement demographic data collection (with consent)",
            "Regular auditing of data sources for bias"
        ])
        
        recommendations['monitoring'].extend([
            "Establish regular bias monitoring dashboard",
            "Set up automated alerts for significant bias indicators",
            "Conduct quarterly diversity analysis reports"
        ])
        
        return recommendations

    def generate_bias_report(self, results: List[Dict], resumes_data: List[Dict]) -> str:
        """Generate a comprehensive bias analysis report."""
        
        analysis = self.analyze_diversity_metrics(results, resumes_data)
        
        report = []
        report.append("=" * 60)
        report.append("DIVERSITY & BIAS ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary
        summary = analysis['summary']
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 20)
        report.append(f"Total Candidates Analyzed: {summary['total_candidates']}")
        report.append(f"Category Diversity Index: {summary['category_diversity_index']}")
        report.append(f"Gender Diversity Index: {summary['gender_diversity_index']}")
        report.append(f"Overall Assessment: {summary['diversity_assessment']}")
        report.append("")
        
        # Category Analysis
        if 'category_analysis' in analysis:
            category_analysis = analysis['category_analysis']
            report.append("CATEGORY BIAS ANALYSIS")
            report.append("-" * 25)
            
            for category, performance in category_analysis['category_performance'].items():
                report.append(f"{category}: Mean Score = {performance['mean_score']}")
            
            if category_analysis['bias_indicators']['potential_bias']:
                report.append("\n⚠️  POTENTIAL BIAS DETECTED")
                for concern in category_analysis['bias_indicators']['concerns']:
                    report.append(f"  - {concern}")
            report.append("")
        
        # Recommendations
        if 'recommendations' in analysis:
            recommendations = analysis['recommendations']
            report.append("RECOMMENDATIONS")
            report.append("-" * 15)
            
            if recommendations['immediate_actions']:
                report.append("Immediate Actions:")
                for action in recommendations['immediate_actions']:
                    report.append(f"  • {action}")
                report.append("")
            
            if recommendations['algorithm_improvements']:
                report.append("Algorithm Improvements:")
                for improvement in recommendations['algorithm_improvements']:
                    report.append(f"  • {improvement}")
        
        return "\n".join(report) 