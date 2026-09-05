"""
@Author :Axin
@Time   :20:24
@Desc   :
"""

from transformers import AutoTokenizer
import pandas as pd

data = pd.read_parquet("./Countdown-Tasks-3to4/data/")

SYSTEM_MESSAGE = (
    "你是一个优秀的助手。你首先在脑海中思考推理过程。"
    "人后为用户提供答案。"
)

USER_TEMPLATE = (
    "使用这些数字 {numbers} ，创建一个等于 {target} 的等式。"
    "你可以使用基本的算术运算 (+, -, *, /)，所有数字都必须使用，"
    "且每个数字只能使用一次，"
    "在 <think> </think> 标签中展示你的思考过程和解题过程，"
    "在 <answer> </answer> 标签中返回最终答案，"
    "例如：<answer> (1 + 2) / 3 </answer>。"
)

tokenizer = AutoTokenizer.from_pretrained(r"D:\class_sources\LLM\model\Qwen3-0.6B-Base")

tokenizer.chat_template = r"""
{% for message in messages %}
{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
{% endfor %}
{% if add_generation_prompt %}
{{ '<|im_start|>assistant\n让我一步一步来解决问题。\n<think>\n' }}
{% endif %}
"""

messages = [
    {"role": "system", "content": SYSTEM_MESSAGE},
    {"role": "user", "content": USER_TEMPLATE.format(numbers=data.iloc[0]["nums"],
                                                     target=data.iloc[0]["target"],)}
]

input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)["input_ids"]

print(tokenizer.decode(input_ids))