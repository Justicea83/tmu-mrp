"""
Explainable AI Module for Resume-Job Ranking System

This module provides detailed explanations for ranking decisions, helping users understand
why certain matches were made and how to improve match scores.
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging


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

    def explain_ranking(self, resume: Dict[str, Any], job: Dict[str, Any], 
                       scores: Dict[str, float], weights: Dict[str, float],
                       all_results: List[Dict] = None) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a resume-job ranking.
        
        Args:
            resume: Resume data
            job: Job description data
            scores: Dictionary of individual dimension scores
            weights: Dictionary of dimension weights
            all_results: All ranking results for comparative analysis
            
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