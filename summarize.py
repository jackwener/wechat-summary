"""
summarize.py — AI-powered chat summarization using Claude

Takes OCR-extracted chat text and generates a structured Markdown summary.
"""

import os
from pathlib import Path
from datetime import datetime

import anthropic
from dotenv import load_dotenv

load_dotenv(".env.local")


def create_summary_prompt(chat_text: str) -> str:
    """Create the summarization prompt for Claude."""
    return f"""你是一个微信群聊总结助手。以下是从微信群聊截图中通过 OCR 提取的聊天记录。
请对这些聊天内容进行结构化整理和总结。

## 要求

1. **讨论纪要**（详细整理）：
   - 按时间顺序和话题，**完整地**整理聊天内容
   - 将口语化的表达改写为更通顺、易读的书面语
   - 保留所有有意义的发言，标注发言人
   - 不要过度精简，保持与原文接近的信息量
   - 修正 OCR 识别错误、补充被截断的内容
   - 合并同一人的连续发言，但不要合并不同话题

2. **概要**：一段话概括主要讨论内容

3. **话题讨论**：按主题归类，提炼核心观点

4. **提取关键信息**：
   - 重要决定或结论
   - 分享的链接或资源
   - Action items / 待办事项
   - 有价值的观点或建议

5. **中文输出**：使用中文撰写

## 输出格式

使用 Markdown 格式，结构如下：

# 群聊总结 — [日期]

## 📝 讨论纪要
（按时间顺序整理的完整讨论内容，口语转书面语，保留细节和发言人。
  按话题用三级标题分隔，每个发言人的内容用列表呈现。）

## 📋 概要
（一段话概括今天的主要讨论内容）

## 💬 话题讨论

### 话题 1: [话题名称]
- **[发言人A]**: 观点/内容
- **[发言人B]**: 观点/内容
- ...

### 话题 2: [话题名称]
...

## ✅ Action Items
- [ ] 待办事项 1 (负责人: xxx)
- [ ] 待办事项 2

## 🔗 分享的资源
- [资源描述](链接)

---

## 聊天记录

```
{chat_text}
```"""


def summarize_chat(
    chat_text: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 8192,
) -> str:
    """
    Generate a structured summary of the chat text using Claude.

    Args:
        chat_text: OCR-extracted chat text
        model: Claude model to use
        max_tokens: Maximum tokens for the response

    Returns:
        Markdown-formatted summary
    """
    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY environment variable is required. "
            "Set it in .env.local or your environment."
        )

    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    client = anthropic.Anthropic(api_key=api_key, **(dict(base_url=base_url) if base_url else {}))

    prompt = create_summary_prompt(chat_text)

    print(f"Sending {len(chat_text)} chars to Claude ({model})...")

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    summary = message.content[0].text
    print(f"Summary generated: {len(summary)} chars")
    print(f"Tokens used: input={message.usage.input_tokens}, output={message.usage.output_tokens}")

    return summary


def save_summary(
    summary: str,
    output_dir: str = "output",
    group_name: str = "群聊",
) -> str:
    """
    Save the summary as a Markdown file.

    Args:
        summary: The generated summary text
        output_dir: Output directory
        group_name: Name of the chat group

    Returns:
        Path to the saved file
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{date_str}_{group_name}.md"
    filepath = out / filename

    filepath.write_text(summary, encoding="utf-8")
    print(f"Summary saved to {filepath}")
    return str(filepath)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize WeChat chat text")
    parser.add_argument("--input", required=True, help="Input text file")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--group", default="群聊", help="Group chat name")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model")
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    summary = summarize_chat(text, model=args.model)
    save_summary(summary, output_dir=args.output_dir, group_name=args.group)
