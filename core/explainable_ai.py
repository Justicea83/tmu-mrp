"""
Explainable AI Module for Resume-Job Ranking System

This module provides detailed explanations for ranking decisions, helping users understand
why certain matches were made and how to improve match scores.
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None


class ExplainableAI:
    """
    Provides explanations and insights for resume-job matching decisions.
    
    Features:
    - Detailed score breakdowns
    - Top contributing factors
    - Improvement recommendations
    - Similar profile analysis
    - Comparative explanations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Thresholds for categorizing scores
        self.score_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'average': 0.4,
            'poor': 0.2
        }
        
        # Weight importance for explanations
        self.dimension_importance = {
            'general': 'overall content similarity',
            'skills': 'technical and professional skills match',
            'experience': 'career level and work history alignment',
            'location': 'geographic compatibility',
            'education': 'educational background and requirements'
        }
        
        # SHAP explainer (will be initialized when needed)
        self.shap_explainer = None
        self.shap_background_data = None

    def explain_ranking(self, resume: Dict[str, Any], job: Dict[str, Any], 
                       scores: Dict[str, float], weights: Dict[str, float],
                       all_results: List[Dict] = None, use_shap: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a resume-job ranking.
        
        Args:
            resume: Resume data
            job: Job description data
            scores: Dictionary of individual dimension scores
            weights: Dictionary of dimension weights
            all_results: All ranking results for comparative analysis
            use_shap: Whether to include SHAP-based explanations
            
        Returns:
            Dict containing detailed explanation
        """
        
        explanation = {
            'match_summary': self._generate_match_summary(scores, weights),
            'score_breakdown': self._explain_score_breakdown(scores, weights),
            'top_strengths': self._identify_top_strengths(scores, weights),
            'improvement_areas': self._identify_improvement_areas(resume, job, scores),
            'detailed_analysis': self._generate_detailed_analysis(resume, job, scores),
            'recommendations': self._generate_recommendations(resume, job, scores),
            'confidence_level': self._calculate_confidence_level(scores)
        }
        
        if all_results:
            explanation['comparative_analysis'] = self._generate_comparative_analysis(
                resume, job, scores, all_results
            )
        
        # Add SHAP explanations if requested and available
        if use_shap and SHAP_AVAILABLE:
            explanation['shap_analysis'] = self.explain_with_shap(
                resume, job, scores, weights, all_results
            )
        
        return explanation

    def _generate_match_summary(self, scores: Dict[str, float], 
                               weights: Dict[str, float]) -> Dict[str, Any]:
        """Generate high-level match summary."""
        
        # Calculate weighted total score
        total_score = sum(scores.get(dim, 0) * weights.get(dim, 0) 
                         for dim in weights.keys())
        total_weight = sum(weights.values())
        final_score = total_score / total_weight if total_weight > 0 else 0
        
        # Categorize match quality
        if final_score >= self.score_thresholds['excellent']:
            match_quality = 'Excellent Match'
            description = 'This candidate is highly suitable for the position'
        elif final_score >= self.score_thresholds['good']:
            match_quality = 'Good Match'
            description = 'This candidate meets most requirements well'
        elif final_score >= self.score_thresholds['average']:
            match_quality = 'Average Match'
            description = 'This candidate has some relevant qualifications'
        else:
            match_quality = 'Poor Match'
            description = 'This candidate may not be suitable for this position'
        
        return {
            'overall_score': round(final_score, 3),
            'match_quality': match_quality,
            'description': description,
            'confidence': self._calculate_confidence_level(scores)
        }

    def _explain_score_breakdown(self, scores: Dict[str, float], 
                                weights: Dict[str, float]) -> Dict[str, Any]:
        """Provide detailed breakdown of how scores were calculated."""
        
        breakdown = {}
        total_weighted_score = 0
        total_weight = sum(weights.values())
        
        for dimension, score in scores.items():
            weight = weights.get(dimension, 0)
            weighted_score = score * weight
            total_weighted_score += weighted_score
            
            # Categorize individual score
            if score >= self.score_thresholds['excellent']:
                category = 'Excellent'
            elif score >= self.score_thresholds['good']:
                category = 'Good'
            elif score >= self.score_thresholds['average']:
                category = 'Average'
            else:
                category = 'Poor'
            
            breakdown[dimension] = {
                'raw_score': round(score, 3),
                'weight': round(weight, 3),
                'weighted_score': round(weighted_score, 3),
                'contribution_percentage': round((weighted_score / total_weighted_score * 100), 1) if total_weighted_score > 0 else 0,
                'category': category,
                'description': self.dimension_importance.get(dimension, f'{dimension} matching')
            }
        
        return {
            'dimensions': breakdown,
            'total_weighted_score': round(total_weighted_score, 3),
            'final_score': round(total_weighted_score / total_weight, 3) if total_weight > 0 else 0
        }

    def _identify_top_strengths(self, scores: Dict[str, float], 
                               weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify the top 3 strengths of this match."""
        
        # Calculate weighted scores for ranking
        weighted_scores = []
        for dimension, score in scores.items():
            weight = weights.get(dimension, 0)
            weighted_score = score * weight
            weighted_scores.append({
                'dimension': dimension,
                'score': score,
                'weighted_score': weighted_score,
                'description': self.dimension_importance.get(dimension, f'{dimension} matching')
            })
        
        # Sort by weighted score and take top 3
        top_strengths = sorted(weighted_scores, key=lambda x: x['weighted_score'], reverse=True)[:3]
        
        # Format for explanation
        strengths = []
        for strength in top_strengths:
            if strength['score'] >= self.score_thresholds['average']:  # Only include decent scores
                strengths.append({
                    'area': strength['dimension'].title(),
                    'score': round(strength['score'], 3),
                    'description': strength['description'],
                    'impact': 'High' if strength['score'] >= self.score_thresholds['good'] else 'Medium'
                })
        
        return strengths

    def _identify_improvement_areas(self, resume: Dict[str, Any], job: Dict[str, Any], 
                                   scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify areas where the match could be improved."""
        
        improvement_areas = []
        
        for dimension, score in scores.items():
            if score < self.score_thresholds['good']:  # Areas needing improvement
                suggestions = self._generate_dimension_suggestions(dimension, resume, job, score)
                
                improvement_areas.append({
                    'area': dimension.title(),
                    'current_score': round(score, 3),
                    'impact_level': 'High' if score < self.score_thresholds['poor'] else 'Medium',
                    'suggestions': suggestions
                })
        
        # Sort by impact (lowest scores first)
        return sorted(improvement_areas, key=lambda x: x['current_score'])

    def _generate_dimension_suggestions(self, dimension: str, resume: Dict[str, Any], 
                                      job: Dict[str, Any], score: float) -> List[str]:
        """Generate specific suggestions for improving a dimension score."""
        
        suggestions = []
        
        if dimension == 'skills':
            suggestions.extend([
                "Consider highlighting more technical skills that match the job requirements",
                "Add specific tools, technologies, or methodologies mentioned in the job description",
                "Include relevant certifications or training programs"
            ])
        
        elif dimension == 'experience':
            suggestions.extend([
                "Emphasize work experience that closely matches the job level and responsibilities",
                "Include more details about achievements and impacts in similar roles",
                "Highlight leadership or project management experience if relevant"
            ])
        
        elif dimension == 'general':
            suggestions.extend([
                "Align resume language more closely with job description terminology",
                "Include industry-specific keywords and concepts",
                "Better emphasize how your background fits the company's needs"
            ])
        
        elif dimension == 'location':
            suggestions.extend([
                "Consider relocating or highlighting remote work capabilities",
                "Mention willingness to travel if applicable",
                "Emphasize local market knowledge if relevant"
            ])
        
        elif dimension == 'education':
            suggestions.extend([
                "Highlight relevant coursework or academic projects",
                "Include continuing education or professional development",
                "Emphasize how educational background applies to the role"
            ])
        
        return suggestions

    def _generate_detailed_analysis(self, resume: Dict[str, Any], job: Dict[str, Any], 
                                   scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate detailed analysis of the match."""
        
        try:
            # Extract key information
            resume_info = self._extract_resume_highlights(resume)
            job_info = self._extract_job_highlights(job)
            
            return {
                'resume_highlights': resume_info,
                'job_requirements': job_info,
                'alignment_analysis': self._analyze_alignment(resume_info, job_info),
                'risk_factors': self._identify_risk_factors(scores),
                'success_indicators': self._identify_success_indicators(scores)
            }
            
        except Exception as e:
            self.logger.warning(f"Error generating detailed analysis: {e}")
            return {'error': 'Could not generate detailed analysis'}

    def _extract_resume_highlights(self, resume: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key highlights from resume."""
        
        highlights = {
            'category': resume.get('Category', 'Unknown'),
            'key_skills': [],
            'experience_level': 'Unknown',
            'education': []
        }
        
        try:
            if isinstance(resume.get('parsed_json'), str):
                parsed_data = json.loads(resume['parsed_json'])
                
                # Extract skills
                skills = parsed_data.get('skills', [])
                if isinstance(skills, list):
                    highlights['key_skills'] = [skill.get('name', '') for skill in skills[:5]]
                
                # Extract education
                education = parsed_data.get('education', [])
                if isinstance(education, list):
                    highlights['education'] = [
                        f"{edu.get('studyType', '')} in {edu.get('area', '')}" 
                        for edu in education[:3]
                    ]
                
                # Estimate experience level
                work_experience = parsed_data.get('work', [])
                if isinstance(work_experience, list):
                    highlights['experience_level'] = f"~{len(work_experience)} positions"
                    
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.warning(f"Error parsing resume data: {e}")
        
        return highlights

    def _extract_job_highlights(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key highlights from job description."""
        
        return {
            'position': job.get('Position', 'Unknown'),
            'company': job.get('Company Name', 'Unknown'),
            'experience_required': job.get('Exp Years', 'Not specified'),
            'primary_keyword': job.get('Primary Keyword', 'Not specified'),
            'key_requirements': self._extract_key_requirements(job.get('Long Description', ''))
        }

    def _extract_key_requirements(self, job_description: str) -> List[str]:
        """Extract key requirements from job description."""
        
        # Simple keyword extraction - in production, this could be more sophisticated
        common_requirements = [
            'python', 'java', 'javascript', 'react', 'angular', 'sql',
            'machine learning', 'data science', 'aws', 'azure', 'docker',
            'bachelor', 'master', 'degree', 'certification'
        ]
        
        job_lower = job_description.lower()
        found_requirements = [req for req in common_requirements if req in job_lower]
        
        return found_requirements[:5]  # Top 5

    def _analyze_alignment(self, resume_info: Dict, job_info: Dict) -> Dict[str, Any]:
        """Analyze alignment between resume and job."""
        
        alignment = {
            'category_match': resume_info['category'].upper() in job_info['position'].upper(),
            'skill_overlap': len(set(resume_info['key_skills']) & set(job_info['key_requirements'])),
            'experience_level_match': 'assessment_needed'  # Would need more sophisticated logic
        }
        
        return alignment

    def _identify_risk_factors(self, scores: Dict[str, float]) -> List[str]:
        """Identify potential risk factors in the match."""
        
        risks = []
        
        for dimension, score in scores.items():
            if score < self.score_thresholds['poor']:
                risks.append(f"Very low {dimension} match ({score:.2f})")
            elif score < self.score_thresholds['average']:
                risks.append(f"Below average {dimension} match ({score:.2f})")
        
        return risks

    def _identify_success_indicators(self, scores: Dict[str, float]) -> List[str]:
        """Identify positive indicators for match success."""
        
        indicators = []
        
        for dimension, score in scores.items():
            if score >= self.score_thresholds['excellent']:
                indicators.append(f"Excellent {dimension} match ({score:.2f})")
            elif score >= self.score_thresholds['good']:
                indicators.append(f"Strong {dimension} match ({score:.2f})")
        
        return indicators

    def _generate_recommendations(self, resume: Dict[str, Any], job: Dict[str, Any], 
                                scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate actionable recommendations."""
        
        total_score = np.mean(list(scores.values()))
        
        if total_score >= self.score_thresholds['excellent']:
            recommendation = "STRONG RECOMMEND"
            action = "Proceed with interview process"
        elif total_score >= self.score_thresholds['good']:
            recommendation = "RECOMMEND"
            action = "Consider for next round"
        elif total_score >= self.score_thresholds['average']:
            recommendation = "CONDITIONAL"
            action = "Review carefully, may need additional screening"
        else:
            recommendation = "NOT RECOMMENDED"
            action = "Consider other candidates first"
        
        return {
            'overall_recommendation': recommendation,
            'suggested_action': action,
            'next_steps': self._suggest_next_steps(total_score, scores),
            'interview_focus_areas': self._suggest_interview_focus(scores)
        }

    def _suggest_next_steps(self, total_score: float, scores: Dict[str, float]) -> List[str]:
        """Suggest next steps based on score analysis."""
        
        steps = []
        
        if total_score >= self.score_thresholds['good']:
            steps.append("Schedule initial screening interview")
            steps.append("Verify key skills and experience claims")
        elif total_score >= self.score_thresholds['average']:
            steps.append("Conduct detailed phone screening")
            steps.append("Assess cultural fit and motivation")
            steps.append("Consider skills assessment if technical role")
        else:
            steps.append("Review other candidates first")
            steps.append("Consider for future opportunities if improved")
        
        return steps

    def _suggest_interview_focus(self, scores: Dict[str, float]) -> List[str]:
        """Suggest areas to focus on during interviews."""
        
        focus_areas = []
        
        # Focus on areas that need verification
        for dimension, score in scores.items():
            if score >= self.score_thresholds['good']:
                focus_areas.append(f"Verify and deep-dive into {dimension} strengths")
            elif score < self.score_thresholds['average']:
                focus_areas.append(f"Assess {dimension} gaps and potential for growth")
        
        return focus_areas[:4]  # Top 4 focus areas

    def _calculate_confidence_level(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Calculate confidence level in the matching decision."""
        
        score_variance = np.var(list(scores.values()))
        mean_score = np.mean(list(scores.values()))
        
        # High variance suggests inconsistent match across dimensions
        if score_variance < 0.05:
            confidence = 'High'
            description = 'Consistent performance across all dimensions'
        elif score_variance < 0.15:
            confidence = 'Medium'
            description = 'Some variation across dimensions'
        else:
            confidence = 'Low'
            description = 'Significant variation across dimensions - review carefully'
        
        return {
            'level': confidence,
            'description': description,
            'score_variance': round(score_variance, 4),
            'mean_score': round(mean_score, 3)
        }

    def _generate_comparative_analysis(self, resume: Dict[str, Any], job: Dict[str, Any], 
                                     scores: Dict[str, float], 
                                     all_results: List[Dict]) -> Dict[str, Any]:
        """Generate comparative analysis against other candidates."""
        
        try:
            current_total = np.mean(list(scores.values()))
            
            # Calculate percentile ranking
            all_scores = [np.mean([r.get('general_score', 0), r.get('skills_score', 0), 
                                  r.get('experience_score', 0), r.get('location_score', 0)]) 
                         for r in all_results]
            
            percentile = (sum(1 for score in all_scores if score < current_total) / len(all_scores)) * 100
            
            return {
                'percentile_ranking': round(percentile, 1),
                'candidates_outperformed': sum(1 for score in all_scores if score < current_total),
                'total_candidates': len(all_results),
                'top_performer': current_total >= max(all_scores) if all_scores else False,
                'above_average': current_total > np.mean(all_scores) if all_scores else False
            }
            
        except Exception as e:
            self.logger.warning(f"Error in comparative analysis: {e}")
            return {'error': 'Could not generate comparative analysis'}

    def generate_batch_insights(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Generate insights across all ranking results."""
        
        try:
            if not all_results:
                return {'error': 'No results to analyze'}
            
            # Extract scores for analysis
            general_scores = [r.get('general_score', 0) for r in all_results]
            skills_scores = [r.get('skills_score', 0) for r in all_results]
            experience_scores = [r.get('experience_score', 0) for r in all_results]
            location_scores = [r.get('location_score', 0) for r in all_results]
            
            insights = {
                'score_distribution': {
                    'general': {
                        'mean': round(np.mean(general_scores), 3),
                        'std': round(np.std(general_scores), 3),
                        'min': round(min(general_scores), 3),
                        'max': round(max(general_scores), 3)
                    },
                    'skills': {
                        'mean': round(np.mean(skills_scores), 3),
                        'std': round(np.std(skills_scores), 3),
                        'min': round(min(skills_scores), 3),
                        'max': round(max(skills_scores), 3)
                    },
                    'experience': {
                        'mean': round(np.mean(experience_scores), 3),
                        'std': round(np.std(experience_scores), 3),
                        'min': round(min(experience_scores), 3),
                        'max': round(max(experience_scores), 3)
                    },
                    'location': {
                        'mean': round(np.mean(location_scores), 3),
                        'std': round(np.std(location_scores), 3),
                        'min': round(min(location_scores), 3),
                        'max': round(max(location_scores), 3)
                    }
                },
                'recommendations': {
                    'most_discriminative_dimension': self._find_most_discriminative_dimension({
                        'general': general_scores,
                        'skills': skills_scores,
                        'experience': experience_scores,
                        'location': location_scores
                    }),
                    'candidate_pool_quality': self._assess_candidate_pool_quality(all_results)
                }
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating batch insights: {e}")
            return {'error': 'Could not generate batch insights'}

    def _find_most_discriminative_dimension(self, dimension_scores: Dict[str, List[float]]) -> str:
        """Find which dimension best discriminates between candidates."""
        
        max_variance = 0
        most_discriminative = 'general'
        
        for dimension, scores in dimension_scores.items():
            variance = np.var(scores)
            if variance > max_variance:
                max_variance = variance
                most_discriminative = dimension
        
        return most_discriminative

    def explain_with_shap(self, resume: Dict[str, Any], job: Dict[str, Any], 
                         scores: Dict[str, float], weights: Dict[str, float],
                         all_results: List[Dict] = None) -> Dict[str, Any]:
        """
        Generate SHAP-based explanations for ranking decisions.
        
        Args:
            resume: Resume data
            job: Job description data  
            scores: Individual dimension scores
            weights: Dimension weights
            all_results: All ranking results for background distribution
            
        Returns:
            Dict containing SHAP explanations and visualizations
        """
        
        if not SHAP_AVAILABLE:
            self.logger.warning("SHAP not available. Install with: pip install shap")
            return {'error': 'SHAP not installed'}
        
        try:
            # Prepare data for SHAP
            feature_names = list(scores.keys())
            current_scores = np.array([scores[dim] for dim in feature_names])
            current_weights = np.array([weights[dim] for dim in feature_names])
            
            # Create weighted scoring function
            def weighted_scoring_function(score_matrix):
                """Function that SHAP will explain."""
                if score_matrix.ndim == 1:
                    score_matrix = score_matrix.reshape(1, -1)
                
                weighted_scores = score_matrix * current_weights
                return np.sum(weighted_scores, axis=1) / np.sum(current_weights)
            
            # Initialize background data if not available
            if self.shap_background_data is None and all_results:
                self.shap_background_data = self._prepare_background_data(all_results, feature_names)
            
            # Use background data or create simple baseline
            if self.shap_background_data is not None:
                background_data = self.shap_background_data
            else:
                # Create simple background with average scores
                background_data = np.array([[50.0] * len(feature_names)])  # 50% baseline
            
            # Create SHAP explainer
            explainer = shap.Explainer(weighted_scoring_function, background_data)
            
            # Calculate SHAP values
            shap_values = explainer(current_scores.reshape(1, -1))
            
            # Extract SHAP explanation data
            shap_explanation = {
                'base_value': float(shap_values.base_values[0]),
                'feature_contributions': {},
                'expected_value': float(shap_values.base_values[0]),
                'predicted_value': float(weighted_scoring_function(current_scores)),
                'feature_importance_ranking': []
            }
            
            # Process individual feature contributions
            for i, feature_name in enumerate(feature_names):
                contribution = float(shap_values.values[0][i])
                shap_explanation['feature_contributions'][feature_name] = {
                    'shap_value': contribution,
                    'feature_value': float(current_scores[i]),
                    'weight': float(current_weights[i]),
                    'contribution_magnitude': abs(contribution),
                    'contribution_direction': 'positive' if contribution > 0 else 'negative'
                }
            
            # Rank features by importance (absolute SHAP value)
            feature_importance = [(name, abs(shap_explanation['feature_contributions'][name]['shap_value'])) 
                                for name in feature_names]
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            shap_explanation['feature_importance_ranking'] = [name for name, _ in feature_importance]
            
            # Add interpretations
            shap_explanation['interpretation'] = self._interpret_shap_values(
                shap_explanation, scores, weights
            )
            
            # Add what-if analysis
            shap_explanation['what_if_analysis'] = self._generate_what_if_analysis(
                weighted_scoring_function, current_scores, feature_names, current_weights
            )
            
            return shap_explanation
            
        except Exception as e:
            self.logger.error(f"Error generating SHAP explanations: {e}")
            return {'error': f'SHAP explanation failed: {str(e)}'}

    def _prepare_background_data(self, all_results: List[Dict], feature_names: List[str]) -> np.ndarray:
        """Prepare background data for SHAP from all ranking results."""
        
        background_scores = []
        for result in all_results:
            score_row = []
            for feature in feature_names:
                score_key = f'{feature}_score'
                score_row.append(result.get(score_key, 0.0))
            background_scores.append(score_row)
        
        return np.array(background_scores)

    def _interpret_shap_values(self, shap_explanation: Dict, scores: Dict[str, float], 
                              weights: Dict[str, float]) -> Dict[str, Any]:
        """Provide human-readable interpretation of SHAP values."""
        
        interpretations = {
            'summary': '',
            'key_drivers': [],
            'key_detractors': [],
            'relative_importance': {}
        }
        
        # Sort features by SHAP value magnitude
        contributions = shap_explanation['feature_contributions']
        sorted_features = sorted(contributions.items(), 
                               key=lambda x: abs(x[1]['shap_value']), reverse=True)
        
        # Identify top positive and negative contributors
        positive_contributors = [(name, data) for name, data in sorted_features 
                               if data['shap_value'] > 0]
        negative_contributors = [(name, data) for name, data in sorted_features 
                               if data['shap_value'] < 0]
        
        # Generate summary
        if positive_contributors:
            top_driver = positive_contributors[0]
            interpretations['summary'] = f"The strongest driver of this match is {top_driver[0]} " \
                                       f"(contributing +{top_driver[1]['shap_value']:.3f} to the score)"
        
        # Key drivers (top 3 positive)
        interpretations['key_drivers'] = [
            {
                'dimension': name,
                'shap_contribution': data['shap_value'],
                'raw_score': scores[name],
                'interpretation': f"{name.title()} strongly supports this match "
                                f"(score: {scores[name]:.1f}, SHAP: +{data['shap_value']:.3f})"
            }
            for name, data in positive_contributors[:3]
        ]
        
        # Key detractors (top 3 negative)
        interpretations['key_detractors'] = [
            {
                'dimension': name,
                'shap_contribution': data['shap_value'],
                'raw_score': scores[name],
                'interpretation': f"{name.title()} weakens this match "
                                f"(score: {scores[name]:.1f}, SHAP: {data['shap_value']:.3f})"
            }
            for name, data in negative_contributors[:3]
        ]
        
        # Relative importance (normalized SHAP values)
        total_magnitude = sum(abs(data['shap_value']) for data in contributions.values())
        if total_magnitude > 0:
            interpretations['relative_importance'] = {
                name: {
                    'importance_percentage': abs(data['shap_value']) / total_magnitude * 100,
                    'contribution_type': 'positive' if data['shap_value'] > 0 else 'negative'
                }
                for name, data in contributions.items()
            }
        
        return interpretations

    def _generate_what_if_analysis(self, scoring_function, current_scores: np.ndarray, 
                                  feature_names: List[str], weights: np.ndarray) -> Dict[str, Any]:
        """Generate what-if analysis showing impact of score changes."""
        
        what_if = {
            'score_improvements': {},
            'score_impacts': {},
            'optimization_suggestions': []
        }
        
        current_total = scoring_function(current_scores)[0]
        
        # Analyze impact of improving each dimension
        for i, feature_name in enumerate(feature_names):
            # Test 10-point improvement
            modified_scores = current_scores.copy()
            modified_scores[i] = min(100.0, current_scores[i] + 10)
            new_total = scoring_function(modified_scores)[0]
            improvement_impact = new_total - current_total
            
            what_if['score_improvements'][feature_name] = {
                'current_score': float(current_scores[i]),
                'improved_score': float(modified_scores[i]),
                'total_impact': float(improvement_impact),
                'roi': float(improvement_impact / 10) if improvement_impact > 0 else 0  # Return on investment per point
            }
        
        # Generate optimization suggestions based on ROI
        roi_ranking = sorted(what_if['score_improvements'].items(), 
                           key=lambda x: x[1]['roi'], reverse=True)
        
        for feature_name, analysis in roi_ranking[:3]:  # Top 3 suggestions
            if analysis['roi'] > 0:
                what_if['optimization_suggestions'].append({
                    'dimension': feature_name,
                    'suggestion': f"Improving {feature_name} by 10 points would increase total score by {analysis['total_impact']:.2f}",
                    'roi': analysis['roi'],
                    'priority': 'High' if analysis['roi'] > 0.5 else 'Medium'
                })
        
        return what_if

    def generate_shap_summary(self, all_results: List[Dict], 
                             all_scores: List[Dict[str, float]], 
                             weights: Dict[str, float]) -> Dict[str, Any]:
        """Generate SHAP-based summary across all candidates."""
        
        if not SHAP_AVAILABLE:
            return {'error': 'SHAP not available'}
        
        try:
            feature_names = list(weights.keys())
            
            # Prepare data matrix
            score_matrix = []
            for scores in all_scores:
                score_row = [scores.get(dim, 0.0) for dim in feature_names]
                score_matrix.append(score_row)
            
            score_matrix = np.array(score_matrix)
            weight_array = np.array([weights[dim] for dim in feature_names])
            
            # Create scoring function
            def weighted_scoring_function(scores):
                if scores.ndim == 1:
                    scores = scores.reshape(1, -1)
                weighted = scores * weight_array
                return np.sum(weighted, axis=1) / np.sum(weight_array)
            
            # Create explainer with mean as background
            background = np.mean(score_matrix, axis=0).reshape(1, -1)
            explainer = shap.Explainer(weighted_scoring_function, background)
            
            # Calculate SHAP values for all candidates
            shap_values = explainer(score_matrix)
            
            # Aggregate insights
            summary = {
                'global_feature_importance': {},
                'feature_impact_distribution': {},
                'cohort_insights': {}
            }
            
            # Calculate mean absolute SHAP values (global importance)
            mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
            for i, feature_name in enumerate(feature_names):
                summary['global_feature_importance'][feature_name] = {
                    'mean_absolute_impact': float(mean_abs_shap[i]),
                    'rank': int(np.argsort(-mean_abs_shap)[i]) + 1
                }
            
            # Feature impact distribution
            for i, feature_name in enumerate(feature_names):
                feature_shap_values = shap_values.values[:, i]
                summary['feature_impact_distribution'][feature_name] = {
                    'mean_impact': float(np.mean(feature_shap_values)),
                    'std_impact': float(np.std(feature_shap_values)),
                    'positive_impact_count': int(np.sum(feature_shap_values > 0)),
                    'negative_impact_count': int(np.sum(feature_shap_values < 0)),
                    'neutral_impact_count': int(np.sum(feature_shap_values == 0))
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating SHAP summary: {e}")
            return {'error': f'SHAP summary failed: {str(e)}'}

    def _assess_candidate_pool_quality(self, all_results: List[Dict]) -> str:
        """Assess overall quality of candidate pool."""
        
        total_scores = []
        for result in all_results:
            scores = [result.get('general_score', 0), result.get('skills_score', 0),
                     result.get('experience_score', 0), result.get('location_score', 0)]
            total_scores.append(np.mean(scores))
        
        avg_score = np.mean(total_scores)
        
        if avg_score >= self.score_thresholds['excellent']:
            return 'Excellent - High quality candidate pool'
        elif avg_score >= self.score_thresholds['good']:
            return 'Good - Solid candidate pool with strong matches'
        elif avg_score >= self.score_thresholds['average']:
            return 'Average - Mixed candidate pool, some good matches'
        else:
            return 'Below Average - Consider expanding search criteria' 