"""
@Author :Axin
@Time   :18:43
@Desc   :
"""
"""
用大模型生成一个数学运算式，使nums[num1,num2,num3]中每个数都被使用1次且不重复 ，
运用基本运算法则得到一个式子，式子运算结果得 target：float  
例如：  nums[1,2,3]   target=9  大模型就输出 ： （ 1 + 2 ） * 3 
"""
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Episode:
    """ 存储一条轨迹涉及的所有信息"""
    prompt:str  #提问（问题） / 提示词  input_ids
    text:str    #问题 + 答案 组成的文本  input_ids + answer
    #大模型环境下用自回归生成：prefill decode
    prompt_token_ids:List[list]    #整条回答中 prompt的input_ids
    response_token_ids:List[list]      #生成的回答中的tokens_ids

    is_finished:bool  #生成是否结束
    reward:float #一条轨迹的奖励
    # 这里一条轨迹奖励分为：
    #{
        #"format_reward":
        #"answer_reward":
    #}

    reward_info:Dict[str,float]  #奖励的详细信息



@dataclass
class MiniBatch:
    "每个训练的step的微批次数据信息"
    prompt:List[str]   #一批问题
    numbers:List[List[float]]  #一批问题的nums
    targets:List[float]   #一批问题的目标答案
    prompt_token_ids:List[list]  #一批问题的input_ids


















