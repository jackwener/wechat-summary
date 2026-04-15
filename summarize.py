"""
summarize.py — AI-powered chat summarization using Claude

Takes OCR-extracted chat text and generates a structured Markdown summary.
"""

import os
import re
import time
from pathlib import Path
from datetime import datetime

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

6. **不要在输出末尾附带原始聊天记录**：只输出总结内容，不要把原始聊天记录或代码块附在最后

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

<chat_log>
{chat_text}
</chat_log>"""


def create_chunk_summary_prompt(chat_text: str, chunk_index: int, total_chunks: int) -> str:
    """Create prompt for a single chunk summary."""
    return f"""你是一个微信群聊总结助手。以下是第 {chunk_index}/{total_chunks} 个聊天记录分块。

请输出该分块的结构化摘要，重点保留事实、结论、分歧、行动项、资源链接。
不要臆测，不要补充聊天中不存在的信息。

## 输出格式（Markdown）

### Chunk {chunk_index} 概要
- 2-4 句，说明该分块主要讨论了什么

### Chunk {chunk_index} 关键讨论点
- 按主题列出核心观点，尽量标注发言人

### Chunk {chunk_index} 结论与决定
- 若没有，写“无明确结论”

### Chunk {chunk_index} Action Items
- [ ] 事项 (负责人: xxx / 未明确)

### Chunk {chunk_index} 资源与链接
- 资源名称: 链接（如有）

---

## 分块聊天记录

<chat_log>
{chat_text}
</chat_log>"""


def create_merge_summary_prompt(chunk_summaries: list[str]) -> str:
    """Create prompt to merge chunk summaries into one final summary."""
    joined = "\n\n".join(
        f"## 分块摘要 {index}\n{summary}"
        for index, summary in enumerate(chunk_summaries, start=1)
    )
    return f"""你是一个微信群聊总结助手。下面是多个分块摘要，请合并成最终总结。

要求：
1. 合并重复信息，保留时间脉络和话题结构
2. 冲突观点要并列呈现，不要擅自裁决
3. Action Items 去重并尽量补全负责人
4. 链接与资源去重后按主题归类
5. 输出中文，使用 Markdown

最终输出格式必须是：

# 群聊总结 — [日期]

## 📝 讨论纪要
（按时间顺序整理，按话题分段，保留关键细节和发言人）

## 📋 概要
（一段话概括）

## 💬 话题讨论
### 话题 1: [话题名称]
- **[发言人]**: 观点

## ✅ Action Items
- [ ] 事项 (负责人: xxx)

## 🔗 分享的资源
- [资源描述](链接)

---

## 分块摘要输入

{joined}
"""


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
    client = _build_anthropic_client()
    prompt = create_summary_prompt(chat_text)
    return _send_prompt(client, prompt, model, max_tokens, label="full summary")


def summarize_chat_in_chunks(
    chat_chunks: list[str],
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 8192,
    chunk_max_tokens: int = 4096,
) -> str:
    """Summarize chunk texts first, then merge into one final summary."""
    usable_chunks = [chunk for chunk in chat_chunks if chunk and chunk.strip()]
    if not usable_chunks:
        raise ValueError("No non-empty chat chunks provided.")
    if len(usable_chunks) == 1:
        return summarize_chat(usable_chunks[0], model=model, max_tokens=max_tokens)

    client = _build_anthropic_client()
    chunk_summaries = []

    for index, chunk_text in enumerate(usable_chunks, start=1):
        prompt = create_chunk_summary_prompt(chunk_text, index, len(usable_chunks))
        chunk_summary = _send_prompt(
            client,
            prompt,
            model,
            chunk_max_tokens,
            label=f"chunk {index}/{len(usable_chunks)}",
        )
        chunk_summaries.append(chunk_summary)

    merge_prompt = create_merge_summary_prompt(chunk_summaries)
    return _send_prompt(client, merge_prompt, model, max_tokens, label="final merge")


def _load_local_env() -> None:
    """Load .env.local when python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except Exception:  # noqa: BLE001
        return
    load_dotenv(".env.local", override=False)


def _build_anthropic_client():
    """Create Anthropic client after loading environment."""
    _load_local_env()
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "Anthropic SDK is not installed. Install dependencies from requirements.txt."
        ) from exc

    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY environment variable is required. "
            "Set it in .env.local or your environment."
        )

    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    return anthropic.Anthropic(api_key=api_key, **(dict(base_url=base_url) if base_url else {}))


def _send_prompt(
    client,
    prompt: str,
    model: str,
    max_tokens: int,
    label: str,
    max_retries: int = 3,
) -> str:
    """Send prompt to Claude and return concatenated text blocks, with retry on transient errors."""
    import anthropic

    print(f"Sending {len(prompt)} chars to Claude ({model}) for {label}...")

    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text_blocks = [block.text for block in message.content if hasattr(block, "text")]
            summary = "\n".join(text_blocks).strip()
            if not summary:
                raise RuntimeError(f"Claude returned empty response for {label}.")
            print(f"{label} generated: {len(summary)} chars")
            print(f"Tokens used: input={message.usage.input_tokens}, output={message.usage.output_tokens}")
            return summary

        except anthropic.BadRequestError as exc:
            # Content filter (BigModel error 1301) or malformed request — not retryable.
            err_str = str(exc)
            if "1301" in err_str or "unsafe" in err_str.lower() or "sensitive" in err_str.lower():
                print(f"  ⚠️  Content filter triggered for {label}: {exc}. Returning placeholder.")
                return f"[内容被过滤，无法生成摘要 — {label}]"
            raise RuntimeError(f"Claude API bad request for {label}: {exc}") from exc

        except (
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.InternalServerError,
        ) as exc:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)  # 2s, 4s
                print(f"  ⚠️  Transient API error ({type(exc).__name__}): {exc}. Retrying in {wait}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Claude API failed after {max_retries} attempts for {label}: {exc}") from exc


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

    safe_group = _sanitize_filename_component(group_name)
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{date_str}_{safe_group}.md"
    filepath = out / filename

    filepath.write_text(summary, encoding="utf-8")
    if safe_group != group_name:
        print(f"Group name sanitized for filename: {safe_group}")
    print(f"Summary saved to {filepath}")
    return str(filepath)


def _sanitize_filename_component(value: str) -> str:
    """Sanitize user input for safe cross-platform filenames."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip().strip(".")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned[:80]
    return cleaned or "group"


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
