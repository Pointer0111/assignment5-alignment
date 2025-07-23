import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    """
    对提示和输出字符串进行分词，并构建一个掩码，掩码在回复（response）token处为1，其余（提示或padding）为0。

    参数:
        prompt_strs (list[str]): 提示字符串列表。
        output_strs (list[str]): 输出字符串列表。
        tokenizer (PreTrainedTokenizer): 用于分词的分词器。

    返回:
        dict[str, torch.Tensor]: 令 prompt_and_output_lens 为分词后提示和输出字符串长度的列表。返回的字典包含以下键：
            input_ids (torch.Tensor): 形状为 (batch_size, max(prompt_and_output_lens) - 1)，
                表示分词后的提示和输出字符串，去掉最后一个token。
            labels (torch.Tensor): 形状为 (batch_size, max(prompt_and_output_lens) - 1)，
                表示input_ids右移一位（即去掉第一个token）。
            response_mask (torch.Tensor): 形状为 (batch_size, max(prompt_and_output_lens) - 1)，
                在labels中回复token处为1，其余为0的掩码。
    """
    raise NotImplementedError("Not implemented")

