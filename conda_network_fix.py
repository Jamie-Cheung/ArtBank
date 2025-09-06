#!/usr/bin/env python3
"""
Conda Network Configuration Diagnostic and Fix Tool
"""

import subprocess
import sys
import os
import json
import urllib.request
import urllib.error
from pathlib import Path

def run_command(cmd, capture=True):
    """Run a shell command and return output"""
    try:
        if capture:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.returncode, result.stdout, result.stderr
        else:
            return subprocess.run(cmd, shell=True).returncode, "", ""
    except Exception as e:
        return 1, "", str(e)

def test_url(url, timeout=5):
    """Test if a URL is accessible"""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return True, response.code
    except urllib.error.URLError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def get_conda_config():
    """Get current conda configuration"""
    code, stdout, stderr = run_command("conda config --show")
    if code == 0:
        return stdout
    return None

def test_mirrors():
    """Test various conda mirrors for connectivity"""
    mirrors = {
        "Anaconda Default": "https://repo.anaconda.com/pkgs/main/linux-64/repodata.json",
        "Tsinghua (China)": "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/repodata.json",
        "Alibaba (China)": "https://mirrors.aliyun.com/anaconda/pkgs/main/linux-64/repodata.json",
        "USTC (China)": "https://mirrors.ustc.edu.cn/anaconda/pkgs/main/linux-64/repodata.json",
        "BFSU (China)": "https://mirrors.bfsu.edu.cn/anaconda/pkgs/main/linux-64/repodata.json",
    }
    
    print("\n=== Testing Mirror Connectivity ===\n")
    working_mirrors = []
    
    for name, url in mirrors.items():
        print(f"Testing {name}...", end=" ")
        success, result = test_url(url)
        if success:
            print(f"✓ Working (HTTP {result})")
            working_mirrors.append(name)
        else:
            print(f"✗ Failed ({result})")
    
    return working_mirrors

def fix_condarc():
    """Fix common issues in .condarc file"""
    condarc_path = Path.home() / ".condarc"
    
    if condarc_path.exists():
        print(f"\n=== Checking {condarc_path} ===\n")
        try:
            with open(condarc_path, 'r') as f:
                content = f.read()
                print("Current .condarc content:")
                print("-" * 40)
                print(content)
                print("-" * 40)
                
                # Check for common issues
                if "https//" in content or "http//" in content:
                    print("\n⚠️  Found malformed URLs (missing colon after protocol)")
                    fixed_content = content.replace("https//", "https://").replace("http//", "http://")
                    
                    # Backup original
                    backup_path = condarc_path.with_suffix(f".backup.{os.getpid()}")
                    condarc_path.rename(backup_path)
                    print(f"   Backed up to: {backup_path}")
                    
                    # Write fixed version
                    with open(condarc_path, 'w') as f:
                        f.write(fixed_content)
                    print("   ✓ Fixed malformed URLs in .condarc")
                    
        except Exception as e:
            print(f"Error reading .condarc: {e}")
    else:
        print(f"\nNo .condarc file found at {condarc_path}")

def setup_mirror(mirror_name):
    """Setup a specific mirror configuration"""
    mirror_configs = {
        "tsinghua": [
            "conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/",
            "conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/",
            "conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/",
            "conda config --set show_channel_urls yes"
        ],
        "alibaba": [
            "conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/main/",
            "conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/free/",
            "conda config --add channels https://mirrors.aliyun.com/anaconda/cloud/conda-forge/",
            "conda config --set show_channel_urls yes"
        ],
        "ustc": [
            "conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/",
            "conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/",
            "conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/",
            "conda config --set show_channel_urls yes"
        ],
        "default": [
            "conda config --remove-key channels",
            "conda config --add channels defaults",
            "conda config --set show_channel_urls yes"
        ]
    }
    
    if mirror_name in mirror_configs:
        print(f"\n=== Setting up {mirror_name} mirror ===\n")
        # Clear existing channels first
        run_command("conda config --remove-key channels", capture=False)
        
        for cmd in mirror_configs[mirror_name]:
            print(f"Running: {cmd}")
            code, _, _ = run_command(cmd, capture=False)
            if code != 0:
                print(f"  Warning: Command failed")
        return True
    return False

def main():
    print("=" * 50)
    print("Conda Network Configuration Diagnostic Tool")
    print("=" * 50)
    
    # Check if conda is available
    print("\n1. Checking conda installation...")
    code, stdout, _ = run_command("conda --version")
    if code != 0:
        print("❌ Conda is not installed or not in PATH")
        sys.exit(1)
    print(f"✓ {stdout.strip()}")
    
    # Fix .condarc if needed
    fix_condarc()
    
    # Test mirrors
    working_mirrors = test_mirrors()
    
    if not working_mirrors:
        print("\n❌ No working mirrors found. Please check your internet connection.")
        print("\nPossible issues:")
        print("  1. No internet connection")
        print("  2. Behind a firewall/proxy")
        print("  3. All mirrors are down (unlikely)")
        
        print("\n=== Alternative Solution ===")
        print("You can try using pip with virtualenv instead:")
        print("  python3 -m venv ldm_env")
        print("  source ldm_env/bin/activate")
        print("  pip install --upgrade pip")
        sys.exit(1)
    
    print(f"\n✓ Found {len(working_mirrors)} working mirror(s)")
    
    # Offer to configure
    print("\n=== Configuration Options ===")
    print("\nWould you like to configure conda with a working mirror?")
    print("1. Tsinghua University mirror (recommended for China)")
    print("2. Alibaba Cloud mirror (recommended for China)")
    print("3. USTC mirror (recommended for China)")
    print("4. Default Anaconda channels")
    print("5. Skip configuration")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        mirror_map = {
            "1": "tsinghua",
            "2": "alibaba", 
            "3": "ustc",
            "4": "default"
        }
        
        if choice in mirror_map:
            if setup_mirror(mirror_map[choice]):
                print("\n✓ Configuration complete!")
                print("\nYou can now try:")
                print("  conda create -n ldm python=3.8.10 -y")
        elif choice == "5":
            print("\nSkipping configuration.")
        else:
            print("\nInvalid choice.")
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
    
    # Show current config
    print("\n=== Current Conda Configuration ===")
    run_command("conda config --show channels", capture=False)

if __name__ == "__main__":
    main()