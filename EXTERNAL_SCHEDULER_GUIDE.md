# Fix GitHub Actions Reliability with External Scheduler

Your GitHub Actions scheduled workflow is unreliable (running 50 minutes late!). Here's how to fix it with a **free external service** that triggers your workflow every hour on the dot.

## Quick Fix: cron-job.org (Free)

### Step 1: Create GitHub Personal Access Token
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Name: "AQI Pipeline Trigger"
4. Select scopes: `repo` and `workflow`
5. Copy the token

### Step 2: Set up cron-job.org
1. Go to: https://cron-job.org/en/signup/
2. Create free account
3. Click "CREATE CRONJOB"
4. Fill in:
   - **Title**: AQI Pipeline Trigger
   - **URL**: `https://api.github.com/repos/YOUR_USERNAME/YOUR_REPO/actions/workflows/aqi-pipeline.yml/dispatches`
   - **Schedule**: Every hour (0 * * * *)
   - **Request Method**: POST
   - **Headers**: 
     - `Authorization: token YOUR_GITHUB_TOKEN`
     - `Content-Type: application/json`
   - **Body**: `{"ref": "main"}`
5. Click "CREATE"

### Step 3: Update Your Workflow
Replace your current `.github/workflows/aqi-pipeline.yml` with:

```yaml
name: Hourly AQI Data Pipeline

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
          retention-days: 7
```

## Alternative: UptimeRobot (Also Free)

1. Go to: https://uptimerobot.com/
2. Create free account
3. Add new monitor:
   - Type: HTTP(s)
   - URL: Your GitHub webhook URL
   - Check interval: 1 hour
   - HTTP method: POST
   - Headers: Add your GitHub token

## Why This Works Better

✅ **External service reliability** - cron-job.org/UptimeRobot are dedicated scheduling services  
✅ **Precise timing** - Triggers exactly every hour, not 50 minutes late  
✅ **Better monitoring** - See when triggers happen and if they fail  
✅ **Free** - No cost for basic usage  
✅ **Keeps your existing setup** - Same GitHub Actions, just better triggers  

## Test Your Setup

After setting up, test it manually:
1. Go to your GitHub repository
2. Click "Actions" tab
3. Click "Run workflow" → "Hourly AQI Data Pipeline"
4. Verify it runs successfully

## Monitor Reliability

- **cron-job.org**: Check their dashboard for trigger history
- **GitHub Actions**: Monitor execution times in Actions tab
- **Expected**: Runs every hour on the dot, not 50 minutes late!

This will fix your reliability issues while keeping everything else the same. 