"""
@Author :Axin
@Time   :19:06
@Desc   :
"""
from pathlib import Path
from typing import List

import pandas as pd
from torch.utils.data import Dataset

SYSTEM_MESSAGE = (
    "你是一个有用的助手。你首先在脑海中思考推理过程，"
    "然后为用户提供答案。"
)
# `{numbers}` 和 `{target}` 是占位符，构建训练数据时会被替换
USER_TEMPLATE = (
    "使用这些数字 {numbers}，创建一个等于 {target} 的等式。"
    "你可以使用基本算术运算（+、-、*、/），每个数字只能使用一次。"
    "在 <think> </think> 标签中展示你的解题过程。"
    "并在 <answer> </answer> 标签中返回最终答案，例如 <answer> (1 + 2) / 3 </answer>。"
)

RESPONSE_PROMPT = "让我一步步来解决这个问题。\n<think>"


class CountdownTaskDataset(Dataset):
    """准备训练的数据集"""
    def __init__(self,tokenizer,dataset_path:str,split:str = "train",test_size:int = 100):
        data = pd.read_parquet(Path(dataset_path) / "data")  #-->：DataFrame   # 临时保存读取的数据 下面self.data才会初始化到对象中
        #索引'test_size'  后面的数据用作测试数据
        self.data = (data.iloc[:-test_size] if split == "train" else data.iloc[-test_size:])
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """使用索引去取出一条数据，转换成字典"""
        #{"nums":[1,2,3] , "target":6}
        item = self.data.iloc[index].to_dict()
        item.update(
            #格式化提示词
            self.encode_prefix(
                item["nums"],  #数字列表
                item["target"],  #目标数字
            )
        )
        return item



    def encode_prefix(self,numbers:List[str],target:int):
        """prefix 是 模型的 真正的输入， 也就是 问题 """
        #格式化模板
        user_message = USER_TEMPLATE.format(numbers=numbers, target=target)

        #自定义聊天模板
        self.tokenizer.chat_template = r"""
        {% for message in messages %}
        {{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
        {% endfor %}
        {% if add_generation_prompt %}
        {{ '<|im_start|>assistant\n让我一步一步来解决问题。\n<think>\n' }}
        {% endif %}"""

        #使用聊天模板去格式化输入的问题（提示词）
        prompt =self.tokenizer.apply_chat_template(
            [
                {"role": "system","content": SYSTEM_MESSAGE},
                {"role":"user","content": user_message},
            ],
            add_generation_prompt= True,
        )
        #将问题切分出来
        return {
            "prompt":self.tokenizer.decode(
                prompt["input_ids"],skip_special_tokens=True
            ),
            "prompt_token_ids": prompt["input_ids"],
        }

