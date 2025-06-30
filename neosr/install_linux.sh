#!/bin/sh

# exit on error
set -e

echo "--- starting neosr installation..."

# function to prompt for package installation
prompt_install() {
    package=$1
    echo "--- the package '$package' is required but not installed."
    printf "--- would you like to install it? [y/N] "
    read -r answer
    case "$answer" in
        [Yy]*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# check if git is installed
if ! command -v git >/dev/null 2>&1; then
    if prompt_install "git"; then
        if command -v apt-get >/dev/null 2>&1; then
            sudo apt-get update && sudo apt-get install -y git
        elif command -v dnf >/dev/null 2>&1; then
            sudo dnf install -y git
        elif command -v pacman >/dev/null 2>&1; then
            sudo pacman -Sy git --noconfirm
        else
            printf "-\033[1;31m-- Error: Could not install git. Please install git manually.\033[0m"
            exit 1
        fi
    else
        printf "\033[1;31m--- git is required for installation, exiting.\033[0m"
        exit 1
    fi
fi

# create and move to installation directory
INSTALL_DIR="$PWD/neosr"

# Handle existing installation
if [ -d "$INSTALL_DIR" ]; then
    if [ -d "$INSTALL_DIR/.git" ]; then
        cd "$INSTALL_DIR"
        git pull --autostash
    else
        printf "\033[1;31m--- directory $INSTALL_DIR exists but is not a git repository.\033[0m"
        printf "\033[1;31m--- please remove or rename it and run the script again.\033[0m"
        exit 1
    fi
else
    git clone https://github.com/neosr-project/neosr >/dev/null 2>&1
    cd neosr
fi

# install uv
if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
else
    if prompt_install "curl"; then
        if command -v apt-get >/dev/null 2>&1; then
            sudo apt-get update && sudo apt-get install -y curl
            curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
        elif command -v dnf >/dev/null 2>&1; then
            sudo dnf install -y curl
            curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
        elif command -v pacman >/dev/null 2>&1; then
            sudo pacman -Sy curl --noconfirm
            curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
        else
            printf "\033[1;31m-- error: could not install curl, please install curl or wget manually.\033[0m"
            exit 1
        fi
    else
        printf "\033[1;31m-- either curl or wget is required for installation, exiting.\033[0m"
        exit 1
    fi
fi

uv self update >/dev/null 2>&1
uv cache clean >/dev/null 2>&1
printf "\033[1m--- syncing dependencies (this might take several minutes)...\033[0m\n"
uv sync

# create aliases
echo "--- adding aliases..."
ALIAS_FILE="$HOME/.neosr_aliases"
cat > "$ALIAS_FILE" << 'EOF'
alias neosr-train='uv run --isolated train.py -opt'
alias neosr-test='uv run --isolated test.py -opt'
alias neosr-convert='uv run --isolated convert.py'
alias neosr-update='git pull --autostash && uv self update && uv sync && uv cache prune'
EOF
# add source to shell config files
for SHELL_RC in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile"; do
    if [ -f "$SHELL_RC" ]; then
        if ! grep -q "source $ALIAS_FILE" "$SHELL_RC"; then
            echo "source $ALIAS_FILE" >> "$SHELL_RC"
        fi
    fi
done

printf "\033[1;32m--- neosr installation complete!\033[0m\n\n"
