# Conda Network Connection Fix Guide

## Problem
You're experiencing `HTTP 000 CONNECTION FAILED` errors when trying to create a conda environment. This is typically caused by:
1. Network connectivity issues
2. Malformed URLs in conda configuration
3. Firewall/proxy blocking connections
4. Mirror server issues

## Quick Fix Solutions

### Solution 1: Fix Malformed URLs in .condarc
The error messages show malformed URLs (`https//` instead of `https://`). Check your `.condarc` file:

```bash
# Check if .condarc exists and view its content
cat ~/.condarc

# If you see URLs like "https//mirrors..." (missing colon), fix them:
sed -i 's|https//|https://|g' ~/.condarc
sed -i 's|http//|http://|g' ~/.condarc
```

### Solution 2: Reset Conda Configuration
```bash
# Backup current configuration
cp ~/.condarc ~/.condarc.backup

# Remove all custom channels
conda config --remove-key channels

# Use default channels
conda config --add channels defaults
```

### Solution 3: Use Working Chinese Mirrors

#### Option A: Tsinghua Mirror
```bash
# Clear existing channels
conda config --remove-key channels

# Add Tsinghua mirrors
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes

# Create environment
conda create -n ldm python=3.8.10 -y
```

#### Option B: Alibaba Cloud Mirror
```bash
# Clear existing channels
conda config --remove-key channels

# Add Alibaba mirrors
conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/main/
conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/free/
conda config --add channels https://mirrors.aliyun.com/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes

# Create environment
conda create -n ldm python=3.8.10 -y
```

#### Option C: USTC Mirror
```bash
# Clear existing channels
conda config --remove-key channels

# Add USTC mirrors
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes

# Create environment
conda create -n ldm python=3.8.10 -y
```

### Solution 4: Use Offline Mode
If all mirrors fail, try creating environment with locally cached packages:

```bash
# Use offline mode (only uses locally cached packages)
conda create -n ldm python=3.8 --offline -y

# Then update to specific version when connection is restored
conda activate ldm
conda update python=3.8.10
```

### Solution 5: Alternative - Use virtualenv + pip
If conda continues to fail, use Python's built-in venv:

```bash
# Create virtual environment
python3 -m venv ldm_env

# Activate it
source ldm_env/bin/activate

# Install packages with pip (using Chinese mirror if needed)
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install --upgrade pip
```

## Diagnostic Commands

```bash
# Check current conda configuration
conda config --show

# Test connectivity to a mirror
curl -I https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/repodata.json

# Check for proxy settings
echo $http_proxy
echo $https_proxy

# Clear conda cache if corrupted
conda clean --all
```

## If Behind Corporate Proxy

```bash
# Set proxy for conda
conda config --set proxy_servers.http http://proxy.server:port
conda config --set proxy_servers.https https://proxy.server:port

# Or set environment variables
export http_proxy=http://proxy.server:port
export https_proxy=http://proxy.server:port
```

## Recommended Steps

1. **First**, run the Python diagnostic script:
   ```bash
   python3 /workspace/conda_network_fix.py
   ```

2. **Or**, run the bash fix script:
   ```bash
   bash /workspace/fix_conda_mirrors.sh
   ```

3. **If automated scripts fail**, manually follow Solution 3 above with different mirrors until one works.

4. **As last resort**, use Solution 5 (virtualenv + pip) as an alternative to conda.