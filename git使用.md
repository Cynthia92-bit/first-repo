# Git 使用学习

## 1. 安装 Git

在 WSL Ubuntu 环境中执行以下命令安装 Git：
```
sudo apt update
sudo apt install git
```

验证安装是否成功：
```
git --version
```

看到 `git version 2.43.0` 的输出，说明安装成功。

## 2. 配置用户信息
```
git config --global user.name "cynthia"
git config --global user.email "18717741189@163.com"
```

查看当前全局配置：
```
git config --global --list
```

## 3. 创建本地仓库
```
mkdir my-project
cd my-project
git init
```

执行后会在当前目录下创建一个隐藏的 `.git` 文件夹，表示 Git 仓库已建立。

## 4. 基本工作流

### 4.1 查看状态
```
git status
```

### 4.2 添加文件到暂存区
```
git add 文件名 # 添加单个文件
git add . # 添加当前目录所有改动
```

### 4.3 提交到本地仓库
```
git commit -m "提交说明"
```

### 4.4 查看提交历史
git log --oneline 

## 5. 分支管理

### 5.1 查看分支
```
git branch # 本地分支
git branch -r # 远程分支
```

### 5.2 创建并切换分支
```
git branch feature-1
git checkout feature-1

或者一步完成
git checkout -b feature-1
```

### 5.3 合并分支

首先切换回主分支，然后合并：
```
git checkout main
git merge feature-1
```

## 6. 关联远程仓库（GitHub）

### 6.1 添加远程仓库地址
```
git remote add origin https://github.com/Cynthia92-bit/first-repo.git
```

### 6.2 推送本地分支到远程
```
git push -u origin main
(`-u` 会将本地 `main` 分支与远程 `main` 分支关联，之后只需 `git push` 即可。)
```

### 6.3 拉取远程更新
```
git pull
```

## 7. 常用命令

| 命令 | 说明 |
|------|------|
| `git init` | 初始化仓库 |
| `git add .` | 暂存所有改动 |
| `git commit -m "msg"` | 提交 |
| `git push` | 推送到远程 |
| `git pull` | 拉取远程更新 |
| `git branch` | 查看分支 |
| `git checkout -b 分支名` | 创建并切换分支 |
| `git merge 分支名` | 合并分支 |
| `git log` | 查看历史 |
| `git status` | 查看状态 |

## 8. 遇到的问题与解决

### 问题1：`fatal: not a git repository`

原因：在未初始化的目录中执行了 `git commit` 等命令。  
解决：先 `git init` 或进入正确的仓库目录。

### 问题2：`git push` 失败，提示权限错误

原因：没有配置 SSH 密钥或 HTTPS 密码错误。  
解决：使用 SSH 方式克隆仓库。
git config --global credential.helper store
