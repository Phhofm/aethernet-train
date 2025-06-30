Write-Host "--- starting neosr installation..."

# check admin privileges
function Test-AdminPrivileges {
    return ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
}

# prompt for package installation
function Prompt-Install {
    param(
        [string]$Package
    )
    Write-Host "--- the package '$Package' is required but not installed."
    $answer = Read-Host "--- would you like to install it? [y/N]"
    return $answer -match '^[Yy]'
}

# elevate privileges if needed and restart script
function Start-AdminSession {
    if (-not (Test-AdminPrivileges)) {
        Write-Host "--- requesting administrator privileges for package installation..."
        Start-Process powershell -Verb RunAs -ArgumentList "-ExecutionPolicy Bypass -Command Set-Location '$PWD'; & '$PSCommandPath'"
        exit
    }
}

# install scoop
function Install-Scoop {
    try {
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
        Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression
        return $true
    }
    catch {
        Write-Host "--- failed to install Scoop: $_"
        return $false
    }
}

# check if git is installed
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    if (Prompt-Install "Git") {
        $installed = $false
        
        # try winget first
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            Write-Host "--- attempting to install git using winget..."
            try {
                Start-AdminSession
                winget install -e --id Git.Git
                $installed = $true
            }
            catch {
                Write-Host "--- failed to install git using winget." -ForegroundColor Red
            }
        }
        
        # if winget failed, try scoop
        if (-not $installed) {
            if (Get-Command scoop -ErrorAction SilentlyContinue) {
                Write-Host "--- attempting to install git using scoop..."
                try {
                    scoop install main/git
                    $installed = $true
                }
                catch {
                    Write-Host "--- failed to install git using scoop." -ForegroundColor Red
                }
            }
            # if scoop is not installed, offer to install it
            else {
                Write-Host "--- scoop is not installed."
                if (Prompt-Install "Scoop package manager") {
                    if (Install-Scoop) {
                        Write-Host "--- scoop installed successfully, installing Git..."
                        try {
                            scoop install main/git
                            $installed = $true
                        }
                        catch {
                            Write-Host "--- failed to install Git using scoop." -ForegroundColor Red
                        }
                    }
                }
            }
        }
        
        if ($installed) {
            # refresh PATH
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        }
        else {
            Write-Error "--- failed to install git through any available method. Please install it manually from https://git-scm.com/" -ForegroundColor Red
            exit 1
        }
    }
    else {
        Write-Host "--- git is required for installation, exiting." -ForegroundColor Red 
        exit 1
    }
}

# create and move to installation directory
$INSTALL_DIR = "$(Get-Location)\neosr"
New-Item -ItemType Directory -Force -Path $INSTALL_DIR | Out-Null

# clone repo
git clone https://github.com/neosr-project/neosr
Set-Location neosr

# install uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# refresh PATH
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# update uv and sync dependencies
uv self update > $null 2>&1
Write-Host "--- syncing dependencies (this might take several minutes)..." -ForegroundColor Green
uv cache clean > $null 2>&1
uv sync

# aliases
$PROFILE_CONTENT = @'
function neosr-train { 
    Set-Location "$INSTALL_DIR"
    uv run --isolated train.py -opt $args 
}
function neosr-test { 
    Set-Location "$INSTALL_DIR"
    uv run --isolated test.py -opt $args 
}
function neosr-convert { 
    Set-Location "$INSTALL_DIR"
    uv run --isolated convert.py $args 
}
function neosr-update {
    Set-Location "$INSTALL_DIR"
    git pull --autostash
    uv self update
    uv sync
}
'@

# create or update powershell profile
if (!(Test-Path -Path $PROFILE)) {
    New-Item -ItemType File -Path $PROFILE -Force | Out-Null
}
Add-Content -Path $PROFILE -Value $PROFILE_CONTENT
Write-Host "--- neosr installation complete!" -ForegroundColor Green
