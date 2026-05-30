"""
Fusion 项目 GitHub 推送脚本

使用方法（在你本地电脑运行）：
1. 安装依赖：pip install requests
2. 运行脚本：python push_to_github.py
3. 按提示输入 GitHub token（不会显示在聊天中）

注意：Token 需要 repo 权限（全选）
创建 Token：https://github.com/settings/tokens

作者：朱子瞻
项目：Fusion - 六边形开源大模型
"""

import os
import json
import subprocess
import getpass
import requests
from pathlib import Path


def create_github_repo(token: str, repo_name: str = "fusion-llm", private: bool = False):
    """
    使用 GitHub API 创建仓库
    
    参数：
        token: GitHub Personal Access Token
        repo_name: 仓库名称
        private: 是否私有（默认 False，公开）
    
    返回：
        仓库 URL
    """
    print(f"\n📦 创建 GitHub 仓库：{repo_name}...")
    
    url = "https://api.github.com/user/repos"
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    
    data = {
        "name": repo_name,
        "description": "Fusion - 六边形开源大模型 | 集百家之长，铸最强开源模型",
        "private": private,
        "has_issues": True,
        "has_projects": True,
        "has_wiki": True,
        "auto_init": False,  # 不要自动创建 README
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 201:
        repo_data = response.json()
        repo_url = repo_data["html_url"]
        clone_url = repo_data["clone_url"]
        
        print(f"✅ 仓库创建成功！")
        print(f"   URL: {repo_url}")
        print(f"   克隆 URL (HTTPS): {clone_url}")
        
        return clone_url
        
    elif response.status_code == 422:
        # 仓库已存在
        print(f"⚠️  仓库 {repo_name} 已存在")
        return f"https://github.com/zhan1206/{repo_name}.git"
        
    else:
        print(f"❌ 创建失败：{response.status_code}")
        print(f"   错误信息：{response.text}")
        return None


def push_to_github(repo_url: str, project_dir: str, use_ssh: bool = False):
    """
    推送代码到 GitHub
    
    参数：
        repo_url: 仓库 URL
        project_dir: 项目目录
        use_ssh: 是否使用 SSH（默认 False，使用 HTTPS）
    """
    print(f"\n🚀 推送代码到 GitHub...")
    
    # 切换到项目目录
    os.chdir(project_dir)
    
    # 如果已经设置了 remote，先删除
    subprocess.run(
        ["git", "remote", "remove", "origin"],
        capture_output=True,
    )
    
    # 添加 remote
    if use_ssh:
        # SSH 格式
        ssh_url = repo_url.replace(
            "https://github.com/",
            "git@github.com:",
        )
        remote_url = ssh_url
    else:
        # HTTPS 格式
        remote_url = repo_url
    
    print(f"   Remote URL: {remote_url}")
    
    result = subprocess.run(
        ["git", "remote", "add", "origin", remote_url],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"❌ 添加 remote 失败：{result.stderr}")
        return False
    
    # 推送代码
    print(f"   推送分支：master")
    
    result = subprocess.run(
        ["git", "push", "-u", "origin", "master"],
        capture_output=True,
        text=True,
    )
    
    if result.returncode == 0:
        print(f"✅ 推送成功！")
        print(f"\n🎉 项目已发布：{repo_url.replace('.git', '')}")
        return True
    else:
        print(f"❌ 推送失败：{result.stderr}")
        
        # 如果是 HTTPS 且失败，提示使用 SSH
        if not use_ssh:
            print(f"\n💡 提示：如果 HTTPS 推送失败，可以尝试使用 SSH：")
            print(f"   1. 生成 SSH key：ssh-keygen -t ed25519 -C \"your_email@example.com\"")
            print(f"   2. 添加 SSH key 到 GitHub：https://github.com/settings/keys")
            print(f"   3. 重新运行脚本，输入 'y' 使用 SSH")
        
        return False


def main():
    print("=" * 60)
    print("Fusion 项目 GitHub 推送脚本")
    print("=" * 60)
    
    # 1. 获取项目目录
    project_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\n📂 项目目录：{project_dir}")
    
    # 2. 检查 Git 状态
    os.chdir(project_dir)
    
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    
    if result.stdout:
        print(f"\n⚠️  有未提交的更改：")
        print(result.stdout)
        
        commit = input("\n是否提交这些更改？(y/N): ").strip().lower()
        
        if commit == 'y':
            subprocess.run(["git", "add", "."])
            commit_msg = input("输入提交信息（默认：Update）：") or "Update"
            subprocess.run(["git", "commit", "-m", commit_msg])
            print(f"✅ 已提交")
        else:
            print(f"⚠️  取消推送")
            return
    
    # 3. 获取 GitHub Token（安全输入，不显示）
    print(f"\n🔐 输入 GitHub Personal Access Token")
    print(f"   创建 Token：https://github.com/settings/tokens")
    print(f"   需要权限：repo（全选）")
    print(f"   （输入时不会显示，这是正常的）")
    
    token = getpass.getpass("Token: ")
    
    if not token:
        print(f"❌ Token 不能为空")
        return
    
    # 4. 创建 GitHub 仓库
    repo_name = "fusion-llm"
    repo_url = create_github_repo(token, repo_name, private=False)
    
    if not repo_url:
        print(f"❌ 创建仓库失败")
        return
    
    # 5. 推送代码
    use_ssh = input("\n使用 SSH 推送？(y/N): ").strip().lower() == 'y'
    
    success = push_to_github(repo_url, project_dir, use_ssh=use_ssh)
    
    if not success and not use_ssh:
        # 如果 HTTPS 失败，询问是否尝试 SSH
        retry_ssh = input("\n是否尝试使用 SSH 推送？(y/N): ").strip().lower()
        
        if retry_ssh == 'y':
            # 修改 URL 为 SSH 格式
            ssh_url = repo_url.replace(
                "https://github.com/",
                "git@github.com:",
            )
            success = push_to_github(ssh_url, project_dir, use_ssh=True)
    
    if success:
        print(f"\n🎉 完成！项目已成功推送到 GitHub")
        print(f"   仓库地址：<ADDRESS_REMOVED>
        print(f"   克隆命令：git clone {repo_url.replace('.git', '')}.git")
    else:
        print(f"\n❌ 推送失败，请检查错误信息")


if __name__ == "__main__":
    main()
