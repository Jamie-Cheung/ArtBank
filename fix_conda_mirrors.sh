#!/bin/bash

# Script to fix conda mirror configuration issues

echo "=== Fixing Conda Mirror Configuration ==="
echo ""

# Backup existing conda configuration
echo "1. Backing up current conda configuration..."
if [ -f ~/.condarc ]; then
    cp ~/.condarc ~/.condarc.backup.$(date +%Y%m%d_%H%M%S)
    echo "   Backup created: ~/.condarc.backup.$(date +%Y%m%d_%H%M%S)"
else
    echo "   No existing .condarc file found"
fi

# Clear all existing channels
echo ""
echo "2. Clearing existing channel configuration..."
conda config --remove-key channels 2>/dev/null || true

# Option 1: Use default conda channels (if you have good connection to anaconda.org)
echo ""
echo "3. Setting up channel configuration..."
echo ""
echo "Choose an option:"
echo "  1) Use default Anaconda channels (anaconda.org)"
echo "  2) Use Tsinghua University mirror (China)"
echo "  3) Use Alibaba Cloud mirror (China)"
echo "  4) Use USTC mirror (China)"
echo "  5) Reset to conda defaults"
echo ""
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "Using default Anaconda channels..."
        conda config --add channels defaults
        conda config --set show_channel_urls yes
        ;;
    2)
        echo "Configuring Tsinghua University mirror..."
        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
        conda config --set show_channel_urls yes
        ;;
    3)
        echo "Configuring Alibaba Cloud mirror..."
        conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/main/
        conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/free/
        conda config --add channels https://mirrors.aliyun.com/anaconda/cloud/conda-forge/
        conda config --add channels https://mirrors.aliyun.com/anaconda/cloud/pytorch/
        conda config --set show_channel_urls yes
        ;;
    4)
        echo "Configuring USTC mirror..."
        conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
        conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
        conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
        conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/pytorch/
        conda config --set show_channel_urls yes
        ;;
    5)
        echo "Resetting to conda defaults..."
        rm -f ~/.condarc
        conda config --set show_channel_urls yes
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

# Show current configuration
echo ""
echo "4. Current conda configuration:"
conda config --show channels

# Test connectivity
echo ""
echo "5. Testing connectivity to configured channels..."
conda search python=3.8.10 --info 2>&1 | head -20

echo ""
echo "=== Configuration complete ==="
echo ""
echo "You can now try creating your environment with:"
echo "  conda create -n ldm python=3.8.10 -y"
echo ""
echo "If you still have issues, try:"
echo "  1. Check your internet connection"
echo "  2. Check if you're behind a proxy (configure with: conda config --set proxy_servers.http http://proxy:port)"
echo "  3. Try a different mirror from the options above"
echo "  4. Use pip with a virtual environment as an alternative"