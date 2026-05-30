# Fusion 项目 GitHub 推送脚本（本地执行）
# 作者：朱子瞻
# 项目：Fusion - 六边形开源大模型

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "Fusion 项目 GitHub 推送脚本" -ForegroundColor Cyan
Write-Host "=" * 60

# 1. 检查 Git
Write-Host "`n🔍 检查 Git..." -ForegroundColor Yellow
try {
    $gitVersion = git --version
    Write-Host "✅ Git 已安装：$gitVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Git 未安装，请先安装 Git" -ForegroundColor Red
    exit 1
}

# 2. 进入项目目录
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir
Write-Host "`n📂 项目目录：$projectDir" -ForegroundColor Yellow

# 3. 检查 Git 状态
Write-Host "`n🔍 检查 Git 状态..." -ForegroundColor Yellow
$status = git status --porcelain
if ($status) {
    Write-Host "⚠️  有未提交的更改" -ForegroundColor Yellow
    git status
    
    $commit = Read-Host "`n是否提交更改？(Y/N)"
    if ($commit -eq 'Y' -or $commit -eq 'y') {
        $msg = Read-Host "输入提交信息（默认：Update）"
        if (-not $msg) { $msg = "Update" }
        
        git add .
        git commit -m $msg
        Write-Host "✅ 已提交" -ForegroundColor Green
    }
}

# 4. 创建 GitHub 仓库
Write-Host "`n📦 创建 GitHub 仓库..." -ForegroundColor Yellow
Write-Host "   仓库名：fusion-llm"
Write-Host "   描述：Fusion - 六边形开源大模型"
Write-Host "`n🔐 请输入 GitHub Personal Access Token"
Write-Host "   创建地址：<ADDRESS_REMOVED>
Write-Host "   需要权限：repo（全选）`n"

$token = Read-Host "Token" -AsSecureString
$tokenPlain = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($token)
)

if (-not $tokenPlain) {
    Write-Host "❌ Token 不能为空" -ForegroundColor Red
    exit 1
}

# 调用 GitHub API 创建仓库
$headers = @{
    "Authorization" = "token $tokenPlain"
    "Accept" = "application/vnd.github.v3+json"
}

$body = @{
    "name" = "fusion-llm"
    "description" = "Fusion - 六边形开源大模型 | 集百家之长，铸最强开源模型"
    "private" = $false
    "has_issues" = $true
    "has_projects" = $true
    "has_wiki" = $true
    "auto_init" = $false
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "https://api.github.com/user/repos" `
        -Method Post `
        -Headers $headers `
        -Body $body `
        -ContentType "application/json"
    
    $repoUrl = $response.html_url
    $cloneUrl = $response.clone_url
    
    Write-Host "`n✅ 仓库创建成功！" -ForegroundColor Green
    Write-Host "   URL: $repoUrl" -ForegroundColor Cyan
    Write-Host "   Clone URL: $cloneUrl" -ForegroundColor Cyan
    
} catch {
    if ($_.Exception.Response.StatusCode -eq 422) {
        Write-Host "`n⚠️  仓库 fusion-llm 已存在" -ForegroundColor Yellow
        $cloneUrl = "https://github.com/zhan1206/fusion-llm.git"
    } else {
        Write-Host "`n❌ 创建失败：$($_.Exception.Message)" -ForegroundColor Red
        Write-Host "   请检查 Token 权限" -ForegroundColor Yellow
        exit 1
    }
}

# 5. 推送代码
Write-Host "`n🚀 推送代码到 GitHub..." -ForegroundColor Yellow

# 移除已存在的 remote
git remote remove origin 2>$null

# 添加 remote（使用 HTTPS + Token）
$tokenWithAuth = $tokenPlain
$remoteUrl = $cloneUrl -replace "https://", "https://${tokenWithAuth}@"
git remote add origin $remoteUrl

# 推送
Write-Host "   推送分支：master" -ForegroundColor Yellow
try {
    $pushResult = git push -u origin master 2>&1
    Write-Host "`n✅ 推送成功！" -ForegroundColor Green
    Write-Host "   项目地址：<ADDRESS_REMOVED>
    Write-Host "`n🎉 Fusion 项目已成功发布到 GitHub！" -ForegroundColor Cyan
} catch {
    Write-Host "`n❌ 推送失败：$($_.Exception.Message)" -ForegroundColor Red
    Write-Host "`n💡 可能的解决方案：" -ForegroundColor Yellow
    Write-Host "   1. 使用 SSH 推送（需要配置 SSH key）" -ForegroundColor Yellow
    Write-Host "   2. 手动推送：" -ForegroundColor Yellow
    Write-Host "      git remote add origin https://github.com/zhan1206/fusion-llm.git" -ForegroundColor Gray
    Write-Host "      git push -u origin master" -ForegroundColor Gray
    exit 1
}

# 6. 清理（移除包含 Token 的 remote）
git remote remove origin
git remote add origin "https://github.com/zhan1206/fusion-llm.git"

Write-Host "`n✅ 已清理 remote（移除 Token）" -ForegroundColor Green
Write-Host "`n📜 后续操作：" -ForegroundColor Cyan
Write-Host "   1. 撤销当前 Token（安全考虑）" -ForegroundColor Yellow
Write-Host "      访问：https://github.com/settings/tokens" -ForegroundColor Gray
Write-Host "`n   2. 克隆项目：" -ForegroundColor Yellow
Write-Host "      git clone https://github.com/zhan1206/fusion-llm.git" -ForegroundColor Gray
Write-Host "`n   3. 安装依赖：" -ForegroundColor Yellow
Write-Host "      cd fusion-llm" -ForegroundColor Gray
Write-Host "      pip install -r requirements.txt" -ForegroundColor Gray

Write-Host "`n" + "=" * 60 -ForegroundColor Cyan
Write-Host "完成！" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan

# 提示用户按任意键退出
Write-Host "`n按任意键退出..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
