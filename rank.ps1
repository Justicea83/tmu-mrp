#Requires -Version 5.1

<#
.SYNOPSIS
    Resume-Job Ranking System - Windows PowerShell Version
    
.DESCRIPTION
    This script runs the multi-dimensional resume-job ranking system with 
    configurable parameters for research and analysis purposes.
    
    Enhanced Windows version with parameter validation, dependency checking,
    and automatic environment setup.

.PARAMETER NumJobs
    Number of job descriptions to process (default: 1)
    
.PARAMETER NumResumes  
    Number of resumes to process (default: 3)
    
.PARAMETER ResumeCategories
    Filter by specific resume categories
    Available: INFORMATION-TECHNOLOGY, AUTOMOBILE, HR
    
.PARAMETER ExcludeResumeCategories
    Exclude specific categories from processing
    
.PARAMETER JobKeywords
    Filter jobs by keywords in position titles
    
.PARAMETER GeneralWeight
    Weight for general semantic matching (default: 8.0)
    
.PARAMETER SkillsWeight
    Weight for skills matching (default: 1.0)
    
.PARAMETER ExperienceWeight
    Weight for experience matching (default: 1.0)
    
.PARAMETER LocationWeight
    Weight for location matching (default: 1.0)
    
.PARAMETER EducationWeight
    Weight for education matching (default: 1.0)
    
.PARAMETER TopK
    Number of top matches to return per job (default: 10)
    
.PARAMETER ModelComparison
    Enable multi-model comparison mode
    
.PARAMETER ModelsToCompare
    List of models to compare
    
.PARAMETER ExplainableAI
    Generate detailed explanations for rankings
    
.PARAMETER DiversityAnalysis
    Perform diversity and bias analysis (includes Gaucher et al. 2011 gender bias detection)
    
.PARAMETER LearningToRank
    Use machine learning to improve rankings
    
.PARAMETER LtrModelType
    Learning-to-rank model type (linear, random_forest, gradient_boosting)
    
.PARAMETER CategoryAnalysis
    Enable detailed category performance analysis
    
.PARAMETER Verbose
    Enable verbose logging
    
.PARAMETER CheckDependencies
    Only check dependencies and environment setup
    
.PARAMETER OpenResults
    Automatically open results directory after completion

.EXAMPLE
    .\rank.ps1
    Run with default configuration
    
.EXAMPLE
    .\rank.ps1 -NumJobs 5 -NumResumes 50 -Verbose
    Basic ranking with more data and verbose output
    
.EXAMPLE
    .\rank.ps1 -ResumeCategories "INFORMATION-TECHNOLOGY","HR" -CategoryAnalysis
    Category comparison analysis
    
.EXAMPLE
    .\rank.ps1 -SkillsWeight 3.0 -GeneralWeight 1.0 -ExperienceWeight 1.0 -LocationWeight 0.5
    Custom weights emphasizing skills
    
.EXAMPLE
    .\rank.ps1 -ModelComparison -ModelsToCompare "careerbert","sentence-transformers/all-mpnet-base-v2"
    Model comparison research
    
.EXAMPLE
    .\rank.ps1 -NumJobs 10 -NumResumes 100 -ResumeCategories "INFORMATION-TECHNOLOGY","HR" -CategoryAnalysis -ModelComparison -ExplainableAI -DiversityAnalysis -LearningToRank -Verbose
    Full research setup with all advanced features

.NOTES
    Author: Resume-Job Ranking System
    Version: 2.0
    Requires: Python 3.8+, dependencies from requirements.txt
    
    Enhanced Features:
    - 5-dimensional matching (General, Skills, Experience, Location, Education)
    - Education Matcher with 177 field mappings  
    - Model comparison (CareerBERT vs All-MPNet)
    - SHAP-enhanced Explainable AI
    - Learning-to-Rank with adaptive cross-validation
    - Comprehensive diversity analytics
    - Gaucher et al. (2011) gender bias detection
#>

[CmdletBinding()]
param(
    [int]$NumJobs = 1,
    [int]$NumResumes = 3,
    [string[]]$ResumeCategories = @("INFORMATION-TECHNOLOGY", "HR"),
    [string[]]$ExcludeResumeCategories = @(),
    [string[]]$JobKeywords = @(),
    [double]$GeneralWeight = 8.0,
    [double]$SkillsWeight = 1.0,
    [double]$ExperienceWeight = 1.0,
    [double]$LocationWeight = 1.0,
    [double]$EducationWeight = 1.0,
    [int]$TopK = 10,
    [switch]$ModelComparison,
    [string[]]$ModelsToCompare = @("sentence-transformers/careerbert-jg", "sentence-transformers/all-mpnet-base-v2"),
    [switch]$ExplainableAI,
    [switch]$DiversityAnalysis,
    [switch]$LearningToRank,
    [ValidateSet("linear", "random_forest", "gradient_boosting")]
    [string]$LtrModelType = "gradient_boosting",
    [switch]$CategoryAnalysis,
    [switch]$CheckDependencies,
    [switch]$OpenResults
)

# ================================================================================================
# FUNCTIONS
# ================================================================================================

function Write-Header {
    param([string]$Title)
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host $Title -ForegroundColor Yellow
    Write-Host "=" * 80 -ForegroundColor Cyan
}

function Write-SubHeader {
    param([string]$Title)
    Write-Host "`n$Title" -ForegroundColor Green
    Write-Host "-" * $Title.Length -ForegroundColor Green
}

function Test-PythonInstallation {
    Write-SubHeader "Checking Python Installation"
    
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
            
            # Check version is 3.8+
            $versionMatch = $pythonVersion -match "Python (\d+)\.(\d+)"
            if ($versionMatch) {
                $majorVersion = [int]$matches[1]
                $minorVersion = [int]$matches[2]
                
                if ($majorVersion -ge 3 -and $minorVersion -ge 8) {
                    Write-Host "✓ Python version is compatible (3.8+)" -ForegroundColor Green
                    return $true
                } else {
                    Write-Host "✗ Python version too old. Need 3.8+, found $majorVersion.$minorVersion" -ForegroundColor Red
                    return $false
                }
            }
        }
    } catch {
        Write-Host "✗ Python not found in PATH" -ForegroundColor Red
        Write-Host "  Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
        return $false
    }
    
    return $false
}

function Test-Dependencies {
    Write-SubHeader "Checking Python Dependencies"
    
    $dependencies = @(
        "torch",
        "transformers", 
        "sentence_transformers",
        "pandas",
        "numpy",
        "scikit-learn",
        "spacy",
        "datasets"
    )
    
    $missingDeps = @()
    
    foreach ($dep in $dependencies) {
        try {
            $result = python -c "import $dep; print('OK')" 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✓ $dep" -ForegroundColor Green
            } else {
                Write-Host "✗ $dep" -ForegroundColor Red
                $missingDeps += $dep
            }
        } catch {
            Write-Host "✗ $dep" -ForegroundColor Red
            $missingDeps += $dep
        }
    }
    
    if ($missingDeps.Count -gt 0) {
        Write-Host "`nMissing dependencies: $($missingDeps -join ', ')" -ForegroundColor Red
        Write-Host "Install with: pip install -r requirements.txt" -ForegroundColor Yellow
        return $false
    }
    
    return $true
}

function Test-SpacyModel {
    Write-SubHeader "Checking spaCy Model"
    
    try {
        $result = python -c "import spacy; nlp = spacy.load('en_core_web_lg'); print('OK')" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ spaCy en_core_web_lg model available" -ForegroundColor Green
            return $true
        } else {
            Write-Host "✗ spaCy en_core_web_lg model not found" -ForegroundColor Red
            Write-Host "  Install with: python -m spacy download en_core_web_lg" -ForegroundColor Yellow
            return $false
        }
    } catch {
        Write-Host "✗ Error checking spaCy model" -ForegroundColor Red
        return $false
    }
}

function Test-Datasets {
    Write-SubHeader "Checking Dataset Files"
    
    $requiredFiles = @(
        "datasets\resumes_final.csv",
        "datasets\job_descriptions.csv"
    )
    
    $allFound = $true
    
    foreach ($file in $requiredFiles) {
        if (Test-Path $file) {
            $size = (Get-Item $file).Length / 1MB
            Write-Host "✓ $file ($('{0:N1}' -f $size) MB)" -ForegroundColor Green
        } else {
            Write-Host "✗ $file not found" -ForegroundColor Red
            $allFound = $false
        }
    }
    
    return $allFound
}

function Test-Environment {
    Write-Header "ENVIRONMENT CHECK"
    
    $checks = @(
        (Test-PythonInstallation),
        (Test-Dependencies),
        (Test-SpacyModel),
        (Test-Datasets)
    )
    
    $allPassed = $checks -notcontains $false
    
    if ($allPassed) {
        Write-Host "`n✓ All environment checks passed!" -ForegroundColor Green
    } else {
        Write-Host "`n✗ Some environment checks failed. Please fix the issues above." -ForegroundColor Red
        return $false
    }
    
    return $true
}

function Build-PythonCommand {
    $args = @(
        "runners\rank.py"
        "--num-jobs", $NumJobs
        "--num-resumes", $NumResumes
        "--general-weight", $GeneralWeight
        "--skills-weight", $SkillsWeight
        "--experience-weight", $ExperienceWeight
        "--location-weight", $LocationWeight
        "--education-weight", $EducationWeight
        "--top-k", $TopK
    )
    
    if ($ResumeCategories.Count -gt 0) {
        $args += "--resume-categories"
        $args += $ResumeCategories
    }
    
    if ($ExcludeResumeCategories.Count -gt 0) {
        $args += "--exclude-resume-categories" 
        $args += $ExcludeResumeCategories
    }
    
    if ($JobKeywords.Count -gt 0) {
        $args += "--job-keywords"
        $args += $JobKeywords
    }
    
    if ($ModelComparison) {
        $args += "--model-comparison"
        if ($ModelsToCompare.Count -gt 0) {
            $args += "--models-to-compare"
            $args += $ModelsToCompare
        }
    }
    
    if ($ExplainableAI) { $args += "--explainable-ai" }
    if ($DiversityAnalysis) { $args += "--diversity-analysis" }
    if ($LearningToRank) { 
        $args += "--learning-to-rank"
        $args += "--ltr-model-type", $LtrModelType
    }
    if ($CategoryAnalysis) { $args += "--category-analysis" }
    if ($Verbose) { $args += "--verbose" }
    
    return $args
}

function Show-Configuration {
    Write-Header "CONFIGURATION SUMMARY"
    
    Write-Host "Data Processing:" -ForegroundColor Cyan
    Write-Host "  Jobs: $NumJobs" 
    Write-Host "  Resumes: $NumResumes"
    Write-Host "  Categories: $($ResumeCategories -join ', ')"
    
    Write-Host "`nScoring Weights:" -ForegroundColor Cyan
    Write-Host "  General: $GeneralWeight"
    Write-Host "  Skills: $SkillsWeight" 
    Write-Host "  Experience: $ExperienceWeight"
    Write-Host "  Location: $LocationWeight"
    Write-Host "  Education: $EducationWeight"
    
    Write-Host "`nAdvanced Features:" -ForegroundColor Cyan
    Write-Host "  Model Comparison: $ModelComparison"
    Write-Host "  Explainable AI: $ExplainableAI"
    Write-Host "  Diversity Analysis: $DiversityAnalysis"
    Write-Host "  Learning-to-Rank: $LearningToRank"
    Write-Host "  Category Analysis: $CategoryAnalysis"
    
    if ($ModelComparison) {
        Write-Host "`nModels to Compare:" -ForegroundColor Cyan
        foreach ($model in $ModelsToCompare) {
            Write-Host "  - $model"
        }
    }
}

function Show-Results {
    param([string]$LogsPath)
    
    Write-Header "ANALYSIS RESULTS"
    
    Write-Host "Results saved to: $LogsPath" -ForegroundColor Green
    Write-Host ""
    
    $resultFiles = @(
        @{Pattern = "ranking_results_*.csv"; Description = "Ranking Results"},
        @{Pattern = "diversity_analysis_*.json"; Description = "Diversity Analysis"},
        @{Pattern = "bias_report_*.txt"; Description = "Bias Reports"},
        @{Pattern = "ml_ranking_results_*.csv"; Description = "ML Rankings"},
        @{Pattern = "feature_importance_*.txt"; Description = "Feature Importance"},
        @{Pattern = "explanations_*.json"; Description = "AI Explanations"}
    )
    
    foreach ($fileType in $resultFiles) {
        $files = Get-ChildItem -Path $LogsPath -Name $fileType.Pattern 2>$null
        if ($files) {
            Write-Host "✓ $($fileType.Description): $($files.Count) file(s)" -ForegroundColor Green
        }
    }
    
    Write-Host "`nEnhanced Features Included:" -ForegroundColor Yellow
    $features = @(
        "✓ 5-dimensional matching (General, Skills, Experience, Location, Education)",
        "✓ Education Matcher with 177 field mappings",
        "✓ Model comparison (CareerBERT vs All-MPNet)",
        "✓ SHAP-enhanced Explainable AI", 
        "✓ Learning-to-Rank with adaptive cross-validation",
        "✓ Comprehensive diversity analytics",
        "✓ Gaucher et al. (2011) gender bias detection"
    )
    
    foreach ($feature in $features) {
        Write-Host "  $feature" -ForegroundColor Green
    }
}

# ================================================================================================
# MAIN EXECUTION
# ================================================================================================

try {
    Write-Header "RESUME-JOB RANKING SYSTEM - POWERSHELL VERSION"
    
    # Handle check dependencies only
    if ($CheckDependencies) {
        $envOk = Test-Environment
        if ($envOk) {
            Write-Host "`n✓ Environment is ready!" -ForegroundColor Green
            exit 0
        } else {
            Write-Host "`n✗ Environment setup incomplete." -ForegroundColor Red
            exit 1
        }
    }
    
    # Environment check
    Write-Host "Performing environment check..." -ForegroundColor Cyan
    $envOk = Test-Environment
    
    if (-not $envOk) {
        Write-Host "`nEnvironment check failed. Use -CheckDependencies for detailed diagnostics." -ForegroundColor Red
        exit 1
    }
    
    # Show configuration
    Show-Configuration
    
    # Create logs directory
    $logsPath = "logs"
    if (-not (Test-Path $logsPath)) {
        New-Item -ItemType Directory -Path $logsPath | Out-Null
    }
    
    # Build and execute Python command
    Write-Header "EXECUTING RANKING ANALYSIS"
    Write-Host "Starting multi-dimensional ranking analysis..." -ForegroundColor Cyan
    
    $pythonArgs = Build-PythonCommand
    
    Write-Host "`nRunning: python $($pythonArgs -join ' ')" -ForegroundColor Yellow
    Write-Host ""
    
    & python @pythonArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n" + "=" * 80 -ForegroundColor Green
        Write-Host "RANKING ANALYSIS COMPLETED SUCCESSFULLY" -ForegroundColor Green  
        Write-Host "=" * 80 -ForegroundColor Green
        
        Show-Results -LogsPath $logsPath
        
        if ($OpenResults) {
            Write-Host "`nOpening results directory..." -ForegroundColor Cyan
            Start-Process explorer.exe -ArgumentList $logsPath
        } else {
            Write-Host "`nTo view results: explorer $logsPath" -ForegroundColor Yellow
        }
        
    } else {
        Write-Host "`n" + "=" * 80 -ForegroundColor Red
        Write-Host "RANKING ANALYSIS FAILED" -ForegroundColor Red
        Write-Host "=" * 80 -ForegroundColor Red
        Write-Host ""
        Write-Host "Common solutions:" -ForegroundColor Yellow
        Write-Host "1. Install missing dependencies: pip install -r requirements.txt" 
        Write-Host "2. Download spaCy model: python -m spacy download en_core_web_lg"
        Write-Host "3. Check that datasets exist in the datasets\ directory"
        Write-Host "4. Ensure you have sufficient disk space and memory"
        Write-Host ""
        Write-Host "Run with -CheckDependencies for detailed diagnostics" -ForegroundColor Cyan
        
        exit 1
    }
    
} catch {
    Write-Host "`nUnexpected error occurred:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host $_.ScriptStackTrace -ForegroundColor DarkRed
    exit 1
}

exit 0 