@echo off
echo =====================================
echo CroweTrade GitHub Repository Setup
echo =====================================
echo.

REM Check if gh CLI is installed
where gh >nul 2>&1
if %errorlevel% neq 0 (
    echo GitHub CLI not found. Installing...
    echo.
    echo Please download from: https://cli.github.com/
    echo Or install via winget: winget install --id GitHub.cli
    echo.
    pause
    exit /b 1
)

echo Checking GitHub authentication...
gh auth status >nul 2>&1
if %errorlevel% neq 0 (
    echo You need to authenticate with GitHub first.
    echo Running authentication...
    gh auth login
)

echo.
echo Creating GitHub repository...
gh repo create crowetrade --public --description "Parallel Financial Agent Ecosystem - Quantitative Trading Platform" --clone=false

if %errorlevel% equ 0 (
    echo Repository created successfully!
) else (
    echo Repository might already exist or creation failed.
    echo Continuing with existing repository...
)

echo.
echo Setting remote origin...
git remote remove origin >nul 2>&1
gh repo set-default
for /f "tokens=*" %%i in ('gh api user --jq .login') do set GITHUB_USER=%%i
git remote add origin https://github.com/%GITHUB_USER%/crowetrade.git

echo.
echo Pushing code to GitHub...
git push -u origin master

if %errorlevel% equ 0 (
    echo.
    echo =====================================
    echo SUCCESS! Repository pushed to GitHub
    echo =====================================
    echo.
    echo Your repository is now available at:
    echo https://github.com/%GITHUB_USER%/crowetrade
    echo.
    echo Next steps:
    echo 1. Visit your repository on GitHub
    echo 2. Add a README if needed
    echo 3. Configure GitHub Actions for CI/CD
    echo 4. Set up branch protection rules
    echo.
) else (
    echo.
    echo Push failed. Please check your permissions and try again.
    echo You may need to run: git push -u origin master
)

pause