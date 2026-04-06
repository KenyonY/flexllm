"""Chat 相关的辅助函数"""

from __future__ import annotations

import asyncio
import sys

from .utils import apply_user_template, extract_code_block

# ========== Chat 辅助函数 ==========


def single_chat(
    message,
    model,
    base_url,
    api_key,
    system_prompt,
    temperature,
    max_tokens,
    stream,
    user_template=None,
    thinking=None,
    extract=False,
):
    """单次对话"""

    async def _run():
        from flexllm import LLMClient

        async with LLMClient(model=model, base_url=base_url, api_key=api_key) as client:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            user_content = apply_user_template(message, user_template)
            messages.append({"role": "user", "content": user_content})

            kwargs = {"temperature": temperature, "max_tokens": max_tokens}
            if thinking is not None:
                kwargs["thinking"] = thinking

            if stream and not extract:
                print("Assistant: ", end="", flush=True)
                async for chunk in client.chat_completions_stream(messages, **kwargs):
                    print(chunk, end="", flush=True)
                print()
            else:
                # extract 模式需要完整响应，不能流式
                if stream:
                    full_response = ""
                    async for chunk in client.chat_completions_stream(messages, **kwargs):
                        full_response += chunk
                    result = full_response
                else:
                    result = await client.chat_completions(messages, **kwargs)
                output = str(result)
                if extract:
                    code = extract_code_block(output)
                    if code is not None:
                        print(code)
                    else:
                        print("提示: 回复中未找到代码块，输出原始内容", file=sys.stderr)
                        print(f"Assistant: {output}")
                else:
                    print(f"Assistant: {output}")

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\n[中断]")
    except Exception as e:
        from .errors import ErrorType, cli_error

        cli_error(ErrorType.GENERAL, str(e))


def interactive_chat(
    model,
    base_url,
    api_key,
    system_prompt,
    temperature,
    max_tokens,
    stream,
    user_template=None,
    thinking=None,
):
    """多轮交互对话"""

    async def _run():
        from flexllm import LLMClient

        async with LLMClient(model=model, base_url=base_url, api_key=api_key) as client:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            print("\n多轮对话模式")
            print(f"模型: {model}")
            print(f"服务器: {base_url}")
            print("输入 'quit' 或 Ctrl+C 退出")
            print("-" * 50)

            kwargs = {"temperature": temperature, "max_tokens": max_tokens}
            if thinking is not None:
                kwargs["thinking"] = thinking

            while True:
                try:
                    user_input = input("\nYou: ").strip()

                    if user_input.lower() in ["quit", "exit", "q"]:
                        print("再见！")
                        break

                    if not user_input:
                        continue

                    user_content = apply_user_template(user_input, user_template)
                    messages.append({"role": "user", "content": user_content})

                    if stream:
                        print("Assistant: ", end="", flush=True)
                        full_response = ""
                        async for chunk in client.chat_completions_stream(messages, **kwargs):
                            print(chunk, end="", flush=True)
                            full_response += chunk
                        print()
                        messages.append({"role": "assistant", "content": full_response})
                    else:
                        result = await client.chat_completions(messages, **kwargs)
                        print(f"Assistant: {result}")
                        messages.append({"role": "assistant", "content": result})

                except EOFError:
                    print("\n再见！")
                    break

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\n再见！")
