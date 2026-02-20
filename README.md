# 📱 WeChat Group Chat Summary

自动截图 macOS 微信群聊，通过 OCR 提取文字，再用 Claude AI 生成结构化总结。

> **macOS only** — 依赖 AppleScript 控制微信窗口和 macOS Vision framework 做 OCR。

## ✨ Features

- 🤖 **全自动流程** — 一行命令完成截图 → OCR → AI 总结
- 🔍 **智能导航** — OCR 扫描侧边栏自动定位目标群聊，支持模糊匹配
- 📐 **动态布局检测** — 通过像素分析自适应不同窗口大小和位置，支持深色/浅色模式
- 📸 **滚动截图** — 自动 Page Up 翻页，perceptual hash 去重检测到顶
- 🔤 **高精度 OCR** — 基于 macOS Vision framework，支持中英文混合识别
- 🧹 **跨页去重** — anchor-based 锚点匹配算法，消除翻页截图的重叠内容
- 📝 **结构化总结** — Claude AI 生成讨论纪要、概要、话题讨论、Action Items

## 📋 输出格式

```markdown
# 群聊总结 — 2026-02-18

## 📝 讨论纪要        # 完整的口语→书面语整理
## 📋 概要            # 一段话概括
## 💬 话题讨论        # 按主题归类核心观点
## ✅ Action Items    # 待办事项
## 🔗 分享的资源      # 链接和工具
```

## 🚀 Quick Start

### 前置条件

- **macOS** (需要 AppleScript + Vision framework)
- **微信桌面版** 已登录并打开
- **Python 3.10+**
- **cliclick** — `brew install cliclick`
- **Anthropic API Key** — [获取地址](https://console.anthropic.com/)

### 安装

```bash
git clone https://github.com/your-username/wechat-summary.git
cd wechat-summary

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 配置 API Key
cp .env.example .env.local
# 编辑 .env.local 填入你的 ANTHROPIC_AUTH_TOKEN
```

### 首次运行

macOS 会弹出权限请求，需要授权：
- **辅助功能 (Accessibility)** — 用于控制微信窗口
- **屏幕录制 (Screen Recording)** — 用于截图

### 使用

```bash
# 总结指定群聊（自动导航 + 截图 + OCR + AI 总结）
python main.py --group "群聊名称" --pages 30

# 更多页数（每页约 10-15 条消息）
python main.py --group "AI生产力训练营" --pages 100

# 手动切到目标群后，不指定 --group
python main.py --pages 50
```

总结文件保存在 `output/` 目录，格式为 `YYYY-MM-DD_群名.md`。

## 📖 Usage

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--pages` | 截图页数（每页约 10-15 条消息） | `30` |
| `--group` | 群聊名称（自动导航到该群聊） | 无（使用当前群聊） |
| `--save-text` | 保存 OCR 中间结果到文件 | 不保存 |
| `--model` | Claude 模型 | `claude-sonnet-4-20250514` |
| `--delay` | 翻页间隔（秒） | `0.8` |
| `--chunk-pages` | 分块总结时每块页数（超过后自动 chunking） | `500` |
| `--from-text` | 跳过截图/OCR，直接从文本总结 | — |
| `--from-screenshots` | 跳过截图，从已有截图做 OCR + 总结 | — |

### 跳过步骤（重新生成）

```bash
# 仅重新总结（跳过截图 + OCR）
python main.py --from-text ocr_output.txt --group "群聊名称"

# 从已有截图重新 OCR + 总结
python main.py --from-screenshots ./screenshots/ --group "群聊名称"

# 超大群聊自动分块（每 500 页一个 chunk，总结后再汇总成一个最终 md/pdf）
python main.py --group "群聊名称" --pages 2500 --chunk-pages 500
```

### 单独使用各模块

```bash
# 仅截图
python capture.py --group "群聊名称" --pages 50

# 仅 OCR
python ocr.py --input screenshots/ --output ocr_output.txt

# 仅总结
python summarize.py --input ocr_output.txt --group "群聊名称"
```

## 🏗️ Architecture

```
main.py           # 入口，orchestrates 完整 pipeline + preflight checks
  ├── capture.py  # 截图模块：AppleScript 控制 + 动态布局检测 + 滚动截图
  ├── ocr.py      # OCR 模块：macOS Vision framework + 跨页 anchor 去重
  └── summarize.py # 总结模块：Claude API + 结构化 prompt
```

### 核心流程

```
WeChat Window
     │
     ▼
┌─────────────┐    detect_layout()     ┌──────────────┐
│  Activate   │──────────────────────▶│  Pixel scan   │
│  WeChat     │    sidebar/titlebar/   │  layout info  │
└─────────────┘    inputbox detection  └──────────────┘
     │                                        │
     ▼                                        ▼
┌─────────────┐                      ┌──────────────┐
│  Navigate   │◀─────────────────────│  OCR sidebar │
│  to group   │    find & click      │  match group │
└─────────────┘                      └──────────────┘
     │
     ▼
┌─────────────┐    Page Up scroll    ┌──────────────┐
│  Screenshot │────────────────────▶│  Dedup hash  │
│  chat area  │    repeat N pages   │  stop at top │
└─────────────┘                      └──────────────┘
     │
     ▼
┌─────────────┐    Vision framework  ┌──────────────┐
│  OCR each   │────────────────────▶│  Anchor-based│
│  screenshot │    zh-Hans + en-US  │  dedup merge │
└─────────────┘                      └──────────────┘
     │
     ▼
┌─────────────┐    Claude API        ┌──────────────┐
│  Summarize  │────────────────────▶│  Markdown    │
│  merged text│    structured prompt│  summary     │
└─────────────┘                      └──────────────┘
```

### 动态布局检测 (`detect_layout`)

自动检测微信窗口的 UI 布局，无需硬编码坐标：

- **Sidebar 分隔线** — 多行投票机制，5 条水平扫描线找最大颜色跳变并投票
- **图标栏右边界** — 从左向右扫描首个显著颜色变化
- **标题栏底部** — sidebar 区域上方 15% 找最大颜色跳变
- **输入框顶部** — 聊天区域下方 40% 从底向上找最大颜色跳变

支持深色模式和浅色模式，使用颜色差值绝对量而非方向性假设。

## ⚠️ 注意事项

1. **微信窗口不要被遮挡** — 导航依赖截图做 OCR 定位
2. **首次运行需授权** — macOS 辅助功能和屏幕录制权限
3. **截图期间勿操作** — 自动翻页过程中请勿移动或点击微信
4. **API 费用** — 每次总结调用 Claude API，约 $0.01 - $0.10（取决于消息量）

## 📄 License

MIT
