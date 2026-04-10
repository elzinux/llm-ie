from __future__ import annotations

from typing import Any

from openai import OpenAI

from src.config.settings import API_KEY_ALI

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL_NAME = "qwen-plus"
DEFAULT_MODEL_CONFIG = {
    "api_key": API_KEY_ALI,
    "model_name": DEFAULT_MODEL_NAME,
    "base_url": DEFAULT_BASE_URL,
    "temperature": 0.0,
    "max_tokens": None,
    "timeout": None,
    "enable_thinking": False,
    "extra_body": {},
}


def _resolve_model_config(model_config: dict[str, Any] | None) -> dict[str, Any]:
    """解析并补全模型配置。"""
    if model_config is None:
        return {
            **DEFAULT_MODEL_CONFIG,
            "extra_body": dict(DEFAULT_MODEL_CONFIG.get("extra_body", {})),
        }

    if not isinstance(model_config, dict):
        raise ValueError("model_config 必须是字典或 None。")

    resolved_config = {
        **DEFAULT_MODEL_CONFIG,
        **model_config,
    }
    resolved_config["extra_body"] = dict(resolved_config.get("extra_body", {}))
    return resolved_config


def call_llm(
    user_input: str,
    system_prompt: str,
    model_config: dict[str, Any] | None = None,
) -> str:
    """调用大模型并返回原始文本输出。"""
    if not user_input or not user_input.strip():
        raise ValueError("user_input 不能为空。")

    if not system_prompt or not system_prompt.strip():
        raise ValueError("system_prompt 不能为空。")

    resolved_config = _resolve_model_config(model_config)

    api_key = resolved_config.get("api_key")
    if not api_key:
        raise ValueError("model_config 中缺少必需字段：api_key。")

    model_name = resolved_config.get("model_name", DEFAULT_MODEL_NAME)
    base_url = resolved_config.get("base_url", DEFAULT_BASE_URL)
    temperature = resolved_config.get("temperature")
    max_tokens = resolved_config.get("max_tokens")
    timeout = resolved_config.get("timeout")
    enable_thinking = resolved_config.get("enable_thinking", False)
    extra_body = dict(resolved_config.get("extra_body", {}))

    # 将思考开关统一写入 extra_body。
    extra_body.setdefault("enable_thinking", enable_thinking)

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )

    request_kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        "extra_body": extra_body,
    }

    if temperature is not None:
        request_kwargs["temperature"] = temperature

    if max_tokens is not None:
        request_kwargs["max_tokens"] = max_tokens

    response = client.chat.completions.create(**request_kwargs)
    content = response.choices[0].message.content

    if content is None or not str(content).strip():
        return "模型返回内容为空。"

    return str(content).strip()


def call_extraction_model(
    text: str,
    prompt_template: str,
    model_config: dict[str, Any] | None = None,
    extra_instructions: str | None = None,
) -> dict[str, str]:
    """调用抽取模型执行信息抽取任务。"""
    if not text or not text.strip():
        raise ValueError("text 不能为空。")

    if not prompt_template or not prompt_template.strip():
        raise ValueError("prompt_template 不能为空。")

    final_prompt = prompt_template.strip()
    if extra_instructions and extra_instructions.strip():
        final_prompt = (
            f"{final_prompt}\n\nAdditional instructions:\n{extra_instructions.strip()}"
        )

    raw_output = call_llm(
        user_input=text.strip(),
        system_prompt=final_prompt,
        model_config=model_config,
    )

    return {
        "final_prompt": final_prompt,
        "raw_output": raw_output,
    }


def call_verification_model(
    text: str,
    candidate_output: str,
    verification_template: str,
    model_config: dict[str, Any] | None = None,
    extra_instructions: str | None = None,
) -> dict[str, str]:
    """调用校验模型执行忠实性校验任务。"""
    if not text or not text.strip():
        raise ValueError("text 不能为空。")

    if not candidate_output or not candidate_output.strip():
        raise ValueError("candidate_output 不能为空。")

    if not verification_template or not verification_template.strip():
        raise ValueError("verification_template 不能为空。")

    final_prompt = verification_template.strip()
    if extra_instructions and extra_instructions.strip():
        final_prompt = (
            f"{final_prompt}\n\nAdditional instructions:\n{extra_instructions.strip()}"
        )

    verification_input = (
        "Please verify whether the candidate output is faithful to the source text.\n\n"
        f"[Source Text]\n{text.strip()}\n\n"
        f"[Candidate Output]\n{candidate_output.strip()}"
    )

    raw_output = call_llm(
        user_input=verification_input,
        system_prompt=final_prompt,
        model_config=model_config,
    )

    return {
        "final_prompt": final_prompt,
        "raw_output": raw_output,
    }


def call_reextraction_model(
    user_input: str,
    prompt_template: str,
    model_config: dict[str, Any] | None = None,
    extra_instructions: str | None = None,
) -> dict[str, str]:
    """调用修正模型执行单条记录的重提取。"""
    if not user_input or not user_input.strip():
        raise ValueError("user_input 不能为空。")

    if not prompt_template or not prompt_template.strip():
        raise ValueError("prompt_template 不能为空。")

    final_prompt = prompt_template.strip()
    if extra_instructions and extra_instructions.strip():
        final_prompt = (
            f"{final_prompt}\n\nAdditional instructions:\n{extra_instructions.strip()}"
        )

    raw_output = call_llm(
        user_input=user_input.strip(),
        system_prompt=final_prompt,
        model_config=model_config,
    )

    return {
        "final_prompt": final_prompt,
        "raw_output": raw_output,
    }


if __name__ == "__main__":
    print("开始执行 LLM 客户端示例。")

    user = "我是一个计算机专业的学生，请你为我推荐一本书。"
    prompt = "你是一个热情的 AI 助手，你的任务是帮助用户解决问题。"

    res = call_llm(user, prompt)
    print("模型返回结果：")
    print(res)

    print("示例执行结束。")
