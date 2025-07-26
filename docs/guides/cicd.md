# CI/CD Integration Guide

The SerDes Validation Framework is designed for seamless integration with Continuous Integration and Continuous Deployment (CI/CD) pipelines. This guide covers setup, configuration, and best practices for automated testing.

## Overview

CI/CD integration enables:
- Automated testing on every commit
- Regression detection and prevention
- Quality gates for releases
- Performance monitoring over time
- Cross-platform validation

## Quick Start

### GitHub Actions (Recommended)

Create `.github/workflows/serdes-tests.yml`:

```yaml
name: SerDes Validation Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .
    
    - name: Run tests with mock mode
      env:
        SVF_MOCK_MODE: 1
        PYTHONPATH: ${{ github.workspace }}
      run: |
        python tests/run_tests.py --category all --verbose
    
    - name: Generate coverage report
      run: |
        pip install coverage
        coverage run -m pytest tests/
        coverage xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-${{ matrix.python-version }}
        path: |
          test_reports/
          coverage.xml
```

### Jenkins Pipeline

Create `Jenkinsfile`:

```groovy
pipeline {
    agent any
    
    environment {
        SVF_MOCK_MODE = '1'
        PYTHONPATH = "${WORKSPACE}"
    }
    
    parameters {
        choice(
            name: 'TEST_CATEGORY',
            choices: ['all', 'unit', 'comprehensive', 'integration'],
            description: 'Test category to run'
        )
        booleanParam(
            name: 'INCLUDE_LEGACY',
            defaultValue: false,
            description: 'Include legacy tests'
        )
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Setup Environment') {
            steps {
                sh '''
                    python -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                    pip install -e .
                '''
            }
        }
        
        stage('Lint and Format Check') {
            steps {
                sh '''
                    . venv/bin/activate
                    pip install flake8 black isort
                    flake8 src/ tests/ --max-line-length=100
                    black --check src/ tests/
                    isort --check-only src/ tests/
                '''
            }
        }
        
        stage('Run Tests') {
            steps {
                script {
                    def testCommand = "python tests/run_tests.py --category ${params.TEST_CATEGORY}"
                    if (params.INCLUDE_LEGACY) {
                        testCommand += " --include-legacy"
                    }
                    
                    sh """
                        . venv/bin/activate
                        ${testCommand} --verbose
                    """
                }
            }
        }
        
        stage('Generate Reports') {
            steps {
                sh '''
                    . venv/bin/activate
                    pip install pytest-html pytest-cov
                    python -m pytest tests/ --html=reports/report.html --self-contained-html --cov=src --cov-report=xml
                '''
            }
        }
        
        stage('Performance Regression Check') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    . venv/bin/activate
                    python scripts/performance_regression_check.py
                '''
            }
        }
    }
    
    post {
        always {
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: 'reports',
                reportFiles: 'report.html',
                reportName: 'Test Report'
            ])
            
            publishCoverage adapters: [
                coberturaAdapter('coverage.xml')
            ], sourceFileResolver: sourceFiles('STORE_LAST_BUILD')
        }
        
        failure {
            emailext (
                subject: "Build Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Build failed. Check console output at ${env.BUILD_URL}",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}
```

## Platform-Specific Configurations

### GitLab CI

Create `.gitlab-ci.yml`:

```yaml
stages:
  - test
  - quality
  - deploy

variables:
  SVF_MOCK_MODE: "1"
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

cache:
  paths:
    - .cache/pip/
    - venv/

before_script:
  - python -m venv venv
  - source venv/bin/activate
  - pip install --upgrade pip
  - pip install -r requirements.txt
  - pip install -e .

test:unit:
  stage: test
  script:
    - python tests/run_tests.py --category unit --verbose
  artifacts:
    reports:
      junit: test_reports/junit.xml
    paths:
      - test_reports/
    expire_in: 1 week

test:comprehensive:
  stage: test
  script:
    - python tests/run_tests.py --category comprehensive --verbose
  artifacts:
    reports:
      junit: test_reports/junit.xml
    paths:
      - test_reports/
    expire_in: 1 week

test:integration:
  stage: test
  script:
    - python tests/run_tests.py --category integration --verbose
  allow_failure: true
  artifacts:
    reports:
      junit: test_reports/junit.xml
    paths:
      - test_reports/
    expire_in: 1 week

quality:coverage:
  stage: quality
  script:
    - pip install coverage
    - coverage run -m pytest tests/
    - coverage report
    - coverage xml
  coverage: '/TOTAL.+ ([0-9]{1,3}%)/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

quality:lint:
  stage: quality
  script:
    - pip install flake8 black isort mypy
    - flake8 src/ tests/
    - black --check src/ tests/
    - isort --check-only src/ tests/
    - mypy src/
```

### Azure DevOps

Create `azure-pipelines.yml`:

```yaml
trigger:
- main
- develop

pool:
  vmImage: 'ubuntu-latest'

variables:
  SVF_MOCK_MODE: '1'
  PYTHONPATH: '$(System.DefaultWorkingDirectory)'

strategy:
  matrix:
    Python39:
      python.version: '3.9'
    Python310:
      python.version: '3.10'
    Python311:
      python.version: '3.11'

steps:
- task: UsePythonVersion@0
  inputs:
    versionSpec: '$(python.version)'
  displayName: 'Use Python $(python.version)'

- script: |
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    pip install -e .
  displayName: 'Install dependencies'

- script: |
    python tests/run_tests.py --category all --verbose
  displayName: 'Run tests'
  env:
    SVF_MOCK_MODE: 1

- script: |
    pip install pytest-azurepipelines pytest-cov
    python -m pytest tests/ --cov=src --cov-report=xml --cov-report=html
  displayName: 'Run tests with coverage'

- task: PublishTestResults@2
  condition: succeededOrFailed()
  inputs:
    testResultsFiles: '**/test-*.xml'
    testRunTitle: 'Publish test results for Python $(python.version)'

- task: PublishCodeCoverageResults@1
  inputs:
    codeCoverageTool: Cobertura
    summaryFileLocation: '$(System.DefaultWorkingDirectory)/**/coverage.xml'
    reportDirectory: '$(System.DefaultWorkingDirectory)/**/htmlcov'
```

## Advanced CI/CD Features

### Performance Regression Detection

Create `scripts/performance_regression_check.py`:

```python
#!/usr/bin/env python3
"""
Performance regression detection script for CI/CD
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

def load_performance_baseline():
    """Load performance baseline from previous runs"""
    baseline_file = Path('performance_baseline.json')
    if baseline_file.exists():
        with open(baseline_file) as f:
            return json.load(f)
    return {}

def run_performance_tests():
    """Run performance tests and collect metrics"""
    # Enable mock mode for consistent performance testing
    os.environ['SVF_MOCK_MODE'] = '1'
    
    from serdes_validation_framework.protocols.usb4 import USB4PerformanceBenchmark
    
    benchmark = USB4PerformanceBenchmark()
    results = benchmark.run_performance_tests()
    
    return {
        'timestamp': datetime.now().isoformat(),
        'throughput': results.get('throughput', 0),
        'latency': results.get('latency', 0),
        'memory_usage': results.get('memory_usage', 0),
        'test_duration': results.get('test_duration', 0)
    }

def check_regression(current_metrics, baseline_metrics, threshold=0.1):
    """Check for performance regression"""
    regressions = []
    
    for metric, current_value in current_metrics.items():
        if metric == 'timestamp':
            continue
            
        baseline_value = baseline_metrics.get(metric, 0)
        if baseline_value == 0:
            continue
            
        change_percent = (current_value - baseline_value) / baseline_value
        
        # For latency and test_duration, negative change is good
        if metric in ['latency', 'test_duration']:
            if change_percent > threshold:
                regressions.append({
                    'metric': metric,
                    'current': current_value,
                    'baseline': baseline_value,
                    'change_percent': change_percent * 100
                })
        # For throughput, positive change is good
        elif metric == 'throughput':
            if change_percent < -threshold:
                regressions.append({
                    'metric': metric,
                    'current': current_value,
                    'baseline': baseline_value,
                    'change_percent': change_percent * 100
                })
    
    return regressions

def update_baseline(current_metrics):
    """Update performance baseline"""
    with open('performance_baseline.json', 'w') as f:
        json.dump(current_metrics, f, indent=2)

def main():
    """Main performance regression check"""
    print("Running performance regression check...")
    
    # Load baseline
    baseline = load_performance_baseline()
    
    # Run current tests
    current = run_performance_tests()
    
    # Check for regressions
    regressions = check_regression(current, baseline)
    
    if regressions:
        print("❌ Performance regressions detected:")
        for regression in regressions:
            print(f"  - {regression['metric']}: {regression['change_percent']:.1f}% change")
            print(f"    Current: {regression['current']}")
            print(f"    Baseline: {regression['baseline']}")
        
        # Fail the build
        sys.exit(1)
    else:
        print("✅ No performance regressions detected")
        
        # Update baseline if this is the main branch
        if os.environ.get('CI_COMMIT_REF_NAME') == 'main' or os.environ.get('GITHUB_REF') == 'refs/heads/main':
            update_baseline(current)
            print("📊 Performance baseline updated")

if __name__ == '__main__':
    main()
```

### Multi-Platform Testing

```yaml
# GitHub Actions multi-platform example
name: Multi-Platform Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies (Unix)
      if: runner.os != 'Windows'
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .
    
    - name: Install dependencies (Windows)
      if: runner.os == 'Windows'
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .
    
    - name: Run tests
      env:
        SVF_MOCK_MODE: 1
      run: python tests/run_tests.py --category comprehensive
```

### Docker Integration

Create `Dockerfile.test`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install -e .

# Set environment variables
ENV SVF_MOCK_MODE=1
ENV PYTHONPATH=/app

# Run tests by default
CMD ["python", "tests/run_tests.py", "--category", "all"]
```

Docker Compose for testing:

```yaml
# docker-compose.test.yml
version: '3.8'

services:
  test-runner:
    build:
      context: .
      dockerfile: Dockerfile.test
    environment:
      - SVF_MOCK_MODE=1
      - PYTHONPATH=/app
    volumes:
      - ./test_reports:/app/test_reports
    command: python tests/run_tests.py --category all --verbose
  
  test-unit:
    extends: test-runner
    command: python tests/run_tests.py --category unit
  
  test-comprehensive:
    extends: test-runner
    command: python tests/run_tests.py --category comprehensive
  
  test-integration:
    extends: test-runner
    command: python tests/run_tests.py --category integration
```

## Quality Gates

### Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
  
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100]
  
  - repo: local
    hooks:
      - id: run-tests
        name: Run SerDes Framework Tests
        entry: python tests/run_tests.py --category unit
        language: system
        pass_filenames: false
        always_run: true
        env:
          SVF_MOCK_MODE: "1"
```

### Branch Protection Rules

Configure branch protection in your repository:

```json
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "test (ubuntu-latest, 3.9)",
      "test (ubuntu-latest, 3.10)",
      "test (ubuntu-latest, 3.11)",
      "quality/coverage",
      "quality/lint"
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true
  },
  "restrictions": null
}
```

## Monitoring and Alerting

### Test Result Monitoring

Create `scripts/test_monitoring.py`:

```python
#!/usr/bin/env python3
"""
Test result monitoring and alerting
"""

import json
import requests
import os
from datetime import datetime

def send_slack_notification(webhook_url, message):
    """Send notification to Slack"""
    payload = {
        'text': message,
        'username': 'SerDes CI Bot',
        'icon_emoji': ':robot_face:'
    }
    
    response = requests.post(webhook_url, json=payload)
    return response.status_code == 200

def analyze_test_trends():
    """Analyze test trends and send alerts"""
    # Load test history
    history_file = 'test_history.json'
    if not os.path.exists(history_file):
        return
    
    with open(history_file) as f:
        history = json.load(f)
    
    # Check for concerning trends
    recent_runs = history[-10:]  # Last 10 runs
    failure_rate = sum(1 for run in recent_runs if run['status'] == 'FAILED') / len(recent_runs)
    
    if failure_rate > 0.3:  # More than 30% failure rate
        message = f"⚠️ High test failure rate detected: {failure_rate:.1%} in last 10 runs"
        webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
        if webhook_url:
            send_slack_notification(webhook_url, message)

def record_test_result(status, duration, test_count):
    """Record test result for trend analysis"""
    history_file = 'test_history.json'
    
    # Load existing history
    if os.path.exists(history_file):
        with open(history_file) as f:
            history = json.load(f)
    else:
        history = []
    
    # Add new result
    history.append({
        'timestamp': datetime.now().isoformat(),
        'status': status,
        'duration': duration,
        'test_count': test_count,
        'commit': os.environ.get('GITHUB_SHA', 'unknown'),
        'branch': os.environ.get('GITHUB_REF_NAME', 'unknown')
    })
    
    # Keep only last 100 results
    history = history[-100:]
    
    # Save updated history
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Analyze trends
    analyze_test_trends()

if __name__ == '__main__':
    # This would be called from CI/CD pipeline
    status = os.environ.get('TEST_STATUS', 'UNKNOWN')
    duration = float(os.environ.get('TEST_DURATION', '0'))
    test_count = int(os.environ.get('TEST_COUNT', '0'))
    
    record_test_result(status, duration, test_count)
```

## Best Practices

### 1. Test Organization
- Separate fast unit tests from slower integration tests
- Use appropriate test categories for different CI stages
- Implement proper test isolation

### 2. Performance Optimization
- Cache dependencies between runs
- Use parallel test execution when possible
- Optimize mock operations for speed

### 3. Reliability
- Handle flaky tests appropriately
- Implement proper retry mechanisms
- Use deterministic test data

### 4. Security
- Never commit secrets to version control
- Use secure environment variable management
- Implement proper access controls

### 5. Monitoring
- Track test execution times
- Monitor failure rates and trends
- Set up appropriate alerting

## Troubleshooting

### Common CI/CD Issues

1. **Import Errors in CI**
   ```bash
   # Solution: Ensure PYTHONPATH is set correctly
   export PYTHONPATH=$GITHUB_WORKSPACE
   ```

2. **Mock Mode Not Enabled**
   ```bash
   # Solution: Set environment variable before running tests
   export SVF_MOCK_MODE=1
   ```

3. **Dependency Installation Issues**
   ```bash
   # Solution: Use specific Python version and upgrade pip
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Test Timeouts**
   ```yaml
   # Solution: Increase timeout in CI configuration
   timeout-minutes: 30
   ```

5. **Platform-Specific Failures**
   ```yaml
   # Solution: Use conditional steps
   - name: Install dependencies (Unix)
     if: runner.os != 'Windows'
     run: pip install -r requirements.txt
   ```

### Debug CI/CD Issues

Enable debug logging in CI:

```yaml
- name: Run tests with debug
  env:
    SVF_MOCK_MODE: 1
    SVF_DEBUG: 1
    PYTHONPATH: ${{ github.workspace }}
  run: |
    python -c "import sys; print('Python path:', sys.path)"
    python -c "import os; print('Environment:', {k:v for k,v in os.environ.items() if 'SVF' in k})"
    python tests/run_tests.py --category unit --verbose
```

## Next Steps

- Set up your first CI/CD pipeline using the examples above
- Configure quality gates and branch protection
- Implement performance regression detection
- Set up monitoring and alerting
- Explore [Mock Testing Guide](../tutorials/mock_testing.md) for testing strategies
- Check [Testing Guide](testing.md) for comprehensive testing approaches