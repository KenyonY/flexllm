"""Chat 相关的辅助函数"""

from __future__ import annotations

import asyncio
import sys

from .utils import apply_user_template, extract_code_block

# ========== Chat 辅助函数 ==========


def _is_error_result(result) -> bool:
    """判断返回值是否为失败的 RequestResult（chat_completions 失败时不抛异常）"""
    return hasattr(result, "status") and getattr(result, "status", None) == "error"


def _extract_error_message(result) -> str:
    """从失败的 RequestResult 中提取可读错误信息"""
    data = getattr(result, "data", None)
    if isinstance(data, dict):
        return str(data.get("detail", data.get("error", data)))
    return str(data) if data is not None else "未知错误"


def single_chat(
    message,
    model,
    base_url,
    api_key,
    system_prompt,
    model_params,
    stream,
    user_template=None,
    thinking=None,
    extract=False,
    output_format="text",
):
    """单次对话

    Args:
        model_params: 传给 chat_completions 的参数 dict
            （temperature/max_tokens/response_format/top_p 等）
    """
    import json
    import time

    async def _run():
        from flexllm import LLMClient

        async with LLMClient(model=model, base_url=base_url, api_key=api_key) as client:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            user_content = apply_user_template(message, user_template)
            messages.append({"role": "user", "content": user_content})

            kwargs = dict(model_params)
            if thinking is not None:
                kwargs["thinking"] = thinking

            if output_format == "json":
                t0 = time.perf_counter()
                result = await client.chat_completions(messages, return_usage=True, **kwargs)
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                if _is_error_result(result):
                    _fail(_extract_error_message(result))
                payload = {
                    "content": getattr(result, "content", None)
                    if not isinstance(result, str)
                    else result,
                    "thinking": getattr(result, "reasoning_content", None),
                    "usage": getattr(result, "usage", None),
                    "model": model,
                    "elapsed_ms": elapsed_ms,
                }
                print(json.dumps(payload, ensure_ascii=False))
                return

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
                    if _is_error_result(result):
                        _fail(_extract_error_message(result))
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

    def _fail(error_msg: str):
        from .errors import ErrorType, cli_error

        cli_error(
            ErrorType.NETWORK_ERROR,
            f"LLM 调用失败: {error_msg}",
            context={"model": model, "base_url": base_url},
            suggestion="使用 flexllm test 验证连接，或 flexllm chat --dry-run 检查请求",
            doc="flexllm chat --help",
            retryable=True,
        )

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\n[中断]")
    except Exception as e:
        import typer

        if isinstance(e, typer.Exit):
            # cli_error 已经输出并携带退出码，直接向上传递
            raise
        from .errors import ErrorType, cli_error

        cli_error(
            ErrorType.GENERAL,
            str(e),
            context={
                "exception_type": type(e).__name__,
                "model": model,
                "base_url": base_url,
            },
            suggestion="使用 flexllm test 验证连接，或 flexllm chat --dry-run 检查请求",
            doc="flexllm chat --help",
        )


def interactive_chat(
    model,
    base_url,
    api_key,
    system_prompt,
    model_params,
    stream,
    user_template=None,
    thinking=None,
):
    """多轮交互对话

    Args:
        model_params: 传给 chat_completions 的参数 dict
    """

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

            kwargs = dict(model_params)
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
                        if _is_error_result(result):
                            # 失败：打印错误、回滚本轮 user 消息，不入历史，继续会话
                            messages.pop()
                            print(
                                f"错误: LLM 调用失败: {_extract_error_message(result)}",
                                file=sys.stderr,
                            )
                            continue
                        print(f"Assistant: {result}")
                        messages.append({"role": "assistant", "content": result})

                except EOFError:
                    print("\n再见！")
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    # 单轮失败不退出会话：回滚本轮 user 消息后继续
                    if messages and messages[-1].get("role") == "user":
                        messages.pop()
                    print(f"错误: {type(e).__name__}: {e}", file=sys.stderr)
                    continue

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\n再见！")
