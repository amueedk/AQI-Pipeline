"""
External Scheduler Setup for GitHub Actions
-------------------------------------------
This script sets up a free external service to trigger your GitHub Actions
every hour with much better reliability than GitHub's built-in cron.

Uses cron-job.org (free) to make HTTP requests to GitHub's API every hour.
"""

import requests
import json
import os
from datetime import datetime

def setup_cron_job_org():
    """
    Set up cron-job.org to trigger GitHub Actions every hour.
    This is a free service that's much more reliable than GitHub's cron.
    """
    
    print("=== Setting up External Scheduler for GitHub Actions ===")
    print("This will use cron-job.org (free) to trigger your workflow every hour")
    print()
    
    # Get GitHub details
    repo_owner = input("Enter your GitHub username: ").strip()
    repo_name = input("Enter your repository name: ").strip()
    
    # Create GitHub Personal Access Token
    print("\n=== Step 1: Create GitHub Personal Access Token ===")
    print("1. Go to: https://github.com/settings/tokens")
    print("2. Click 'Generate new token (classic)'")
    print("3. Give it a name like 'AQI Pipeline Trigger'")
    print("4. Select scopes: 'repo' and 'workflow'")
    print("5. Copy the generated token")
    
    github_token = input("\nEnter your GitHub Personal Access Token: ").strip()
    
    # Test the token
    print("\nTesting GitHub token...")
    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    test_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
    response = requests.get(test_url, headers=headers)
    
    if response.status_code != 200:
        print(f"❌ Token test failed: {response.status_code}")
        print("Please check your token and repository details")
        return False
    
    print("✅ GitHub token is valid!")
    
    # Create the webhook URL
    webhook_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/workflows/aqi-pipeline.yml/dispatches"
    
    print(f"\n=== Step 2: Set up cron-job.org ===")
    print("1. Go to: https://cron-job.org/en/signup/")
    print("2. Create a free account")
    print("3. Click 'CREATE CRONJOB'")
    print("4. Fill in these details:")
    print(f"   - Title: AQI Pipeline Trigger")
    print(f"   - URL: {webhook_url}")
    print(f"   - Schedule: Every hour (0 * * * *)")
    print(f"   - Request Method: POST")
    print(f"   - Headers: Authorization: token {github_token}")
    print(f"   - Headers: Content-Type: application/json")
    print(f"   - Body: {{\"ref\": \"main\"}}")
    print("5. Click 'CREATE'")
    
    print("\n=== Step 3: Update GitHub Actions Workflow ===")
    print("Update your .github/workflows/aqi-pipeline.yml to remove the unreliable schedule:")
    
    workflow_content = '''name: Hourly AQI Data Pipeline

on:
  workflow_dispatch:  # Manual trigger
  repository_dispatch:  # External API trigger
    types: [hourly-trigger]

jobs:
  hourly-update:
    runs-on: ubuntu-latest
    timeout-minutes: 10  # Add timeout
    permissions:
      contents: write
    steps:
      - name: Checkout repository code
        uses: actions/checkout@v4

      - name: Set up Python 3.9
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
          cache: 'pip'

      - name: Install Python dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Create necessary directories
        run: |
          mkdir -p logs
          mkdir -p data

      - name: Run automated hourly update script
        run: |
          python automated_hourly_run.py
        env:
          HOPSWORKS_API_KEY: ${{ secrets.HOPSWORKS_API_KEY }}
          OPENWEATHER_API_KEY: ${{ secrets.OPENWEATHER_API_KEY }}
          IQAIR_API_KEY: ${{ secrets.IQAIR_API_KEY }}

      - name: Upload logs and data on failure
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: aqi-logs-${{ github.run_number }}
          path: |
            automated_run.log
            logs/
            data/
          retention-days: 7'''
    
    print("\nCopy this updated workflow content:")
    print("=" * 50)
    print(workflow_content)
    print("=" * 50)
    
    # Save the updated workflow
    workflow_file = ".github/workflows/aqi-pipeline.yml"
    os.makedirs(".github/workflows", exist_ok=True)
    
    with open(workflow_file, 'w') as f:
        f.write(workflow_content)
    
    print(f"\n✅ Updated workflow saved to {workflow_file}")
    
    # Test the webhook
    print("\n=== Step 4: Test the Setup ===")
    test_webhook = input("Would you like to test the webhook now? (y/n): ").strip().lower()
    
    if test_webhook == 'y':
        print("Testing webhook...")
        
        data = {"ref": "main"}
        response = requests.post(webhook_url, headers=headers, json=data)
        
        if response.status_code == 204:
            print("✅ Webhook test successful! Check your GitHub Actions tab.")
        else:
            print(f"❌ Webhook test failed: {response.status_code}")
            print(f"Response: {response.text}")
    
    print("\n=== Setup Complete! ===")
    print("Your GitHub Actions will now be triggered every hour by cron-job.org")
    print("This is much more reliable than GitHub's built-in cron scheduler.")
    print("\nMonitor your workflow at:")
    print(f"https://github.com/{repo_owner}/{repo_name}/actions")
    
    return True

def setup_uptime_robot():
    """
    Alternative: Set up UptimeRobot (also free) to trigger GitHub Actions.
    """
    print("=== Alternative: UptimeRobot Setup ===")
    print("UptimeRobot is another free service that can trigger your workflow.")
    print("\nSteps:")
    print("1. Go to: https://uptimerobot.com/")
    print("2. Create a free account")
    print("3. Add a new monitor:")
    print("   - Type: HTTP(s)")
    print("   - URL: Your GitHub webhook URL")
    print("   - Check interval: 1 hour")
    print("   - HTTP method: POST")
    print("   - Headers: Add your GitHub token")
    
    return True

def main():
    """Main function to set up external scheduler"""
    print("External Scheduler Setup for GitHub Actions")
    print("=" * 50)
    print()
    print("This will set up a free external service to trigger your")
    print("GitHub Actions every hour with much better reliability.")
    print()
    
    choice = input("Choose setup method:\n1. cron-job.org (recommended)\n2. UptimeRobot\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        setup_cron_job_org()
    elif choice == "2":
        setup_uptime_robot()
    else:
        print("Invalid choice. Using cron-job.org...")
        setup_cron_job_org()

if __name__ == "__main__":
    main() 