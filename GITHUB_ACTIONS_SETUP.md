# GitHub Actions Setup Guide

This guide explains how to set up the necessary permissions and secrets for the AQI Pipeline GitHub Actions to work properly.

## Required GitHub Secrets

You need to add the following secrets to your repository:

### 1. API Keys
- `OPENWEATHER_API_KEY` - Your OpenWeather API key
- `IQAIR_API_KEY` - Your IQAir API key  
- `HOPSWORKS_API_KEY` - Your Hopsworks API key

### 2. GitHub Personal Access Token (PAT)
- `GH_PAT` - Personal Access Token for repository write access

## Setting Up the Personal Access Token (PAT)

### Step 1: Create a Personal Access Token

1. Go to your GitHub profile → **Settings**
2. Navigate to **Developer settings** → **Personal access tokens** → **Tokens (classic)**
3. Click **"Generate new token"**
4. Configure the token:
   - **Name**: `GH Actions Push Token`
   - **Expiration**: Choose a suitable expiration (e.g., 90 days)
   - **Scopes**: 
     - ✅ `repo` (Full control of private repositories)
5. Click **Generate token**
6. **Copy the token immediately** (you won't see it again!)

### Step 2: Add the PAT to Repository Secrets

1. Go to your repository → **Settings**
2. Navigate to **Secrets and variables** → **Actions**
3. Click **"New repository secret"**
4. Add the secret:
   - **Name**: `GH_PAT`
   - **Value**: Paste your personal access token
5. Click **Add secret**

## What This Fixes

- **401 Unauthorized Errors**: API keys are now properly passed to GitHub Actions
- **403 Forbidden Errors**: The PAT allows GitHub Actions to push commits to the repository
- **Data Persistence**: AQI validation data can now be committed and pushed back to the repository
- **Permission Issues**: Explicit `contents: write` permissions ensure GitHub Actions can modify the repository

## Workflow Files Updated

The following workflow files have been updated to use the PAT:

- `.github/workflows/aqi-pipeline.yml` - Hourly automated runs
- `.github/workflows/manual-historic-backfill.yml` - Manual historical data backfill

## Testing the Setup

1. After adding all secrets, trigger a manual workflow run
2. Check the logs for the debug messages showing API key status
3. Verify that data files are successfully committed and pushed

## Important Notes

- **Branch Name**: The workflows are configured for the `main` branch. If your default branch is different (e.g., `master`), update `HEAD:main` to `HEAD:your-branch-name` in the git push commands.
- **Permissions**: Both workflows now include explicit `permissions: contents: write` to ensure GitHub Actions can modify the repository.

## Security Notes

- The PAT has minimal required permissions (only `repo` scope)
- The token is stored securely in GitHub Secrets
- The token URL format used is the recommended approach for GitHub Actions 