#!/bin/bash
# 发版脚本：更新 changelog 并创建 tag
# 用法: ./scripts/release.sh v0.4.0

set -e

VERSION=$1

if [ -z "$VERSION" ]; then
    echo "用法: $0 <version>"
    echo "示例: $0 v0.4.0"
    exit 1
fi

# 工作区必须干净：tag 应当指向已提交、已验证的状态。
# 常见误用是忘了先提交 flexllm/__init__.py 的版本号改动，导致 tag 指向旧版本号。
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "错误: 工作区有未提交的改动，请先提交（含版本号 flexllm/__init__.py）"
    git status --short --untracked-files=no
    exit 1
fi

# 版本号必须与 tag 一致
PKG_VERSION="v$(python -c 'import re,pathlib; print(re.search(r"__version__ = \"([^\"]+)\"", pathlib.Path("flexllm/__init__.py").read_text()).group(1))')"
if [ "$PKG_VERSION" != "$VERSION" ]; then
    echo "错误: flexllm/__init__.py 的版本是 $PKG_VERSION，与 tag $VERSION 不一致"
    exit 1
fi

# 检查是否安装 git-cliff
if ! command -v git-cliff &> /dev/null; then
    echo "正在安装 git-cliff..."
    pip install git-cliff -q
fi

echo "生成 CHANGELOG.md..."
git-cliff --tag "$VERSION" -o CHANGELOG.md

echo "提交 changelog 更新..."
git add CHANGELOG.md
# pre-commit 的 end-of-file-fixer 会改写 git-cliff 生成的 CHANGELOG.md，
# 导致首次提交失败退出码非 0。此时文件已被修正，重新 add 再提交一次即可。
# 不加这层保护的话，set -e 会让脚本在打 tag 之前就退出。
git commit -m "chore(release): $VERSION" || {
    echo "（pre-commit 修正了文件，重新提交）"
    git add CHANGELOG.md
    git commit -m "chore(release): $VERSION"
}

echo "创建 tag: $VERSION"
git tag "$VERSION"

echo ""
echo "完成! 执行以下命令推送:"
echo "  git push && git push origin $VERSION"
