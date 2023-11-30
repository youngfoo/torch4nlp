# rm训练
import json
from typing import List
from utils.file_util import read_jsonl


class RMResponse:
    def __init__(self, content:str=None, rm_score:float=None, helpful:float=None, truthful:float=None, saftefy:float=None, source:str=None, comment:str=None):
        self.content = content
        self.rm_score = rm_score
        self.helpful = helpful
        self.truthful = truthful
        self.safety = saftefy
        self.source = source
        self.comment = comment
    
    def to_dict(self):
        return {
            'content': self.content,
            'helpful': self.helpful,
            'truthful': self.truthful,
            'safety': self.safety,
            'rm_score': self.rm_score,
            'source': self.source,
            'comment': self.comment
        }
    
    def __str__(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)


class RMDataItem:
    def __init__(self, batch:str=None, data_type:str=None, task_type:str=None, question:str=None, prompt_id:str=None, prompt:List[str]=None, response:List[RMResponse] = None, vote_label:str=None):
        """
        args:
            time: 标注时间
            batch: 批次号
            data_type: 数据类型
            task_type: 任务树
            question: 问题
            prompt: list of prompt
            prompt_id
            response
        """
        self.batch = batch
        self.ann_time = ann_time
        self.data_type = data_type
        self.task_type = task_type
        self.question = question
        self.prompt_id = prompt_id
        self.prompt = prompt
        self.response = response
    
    def to_dict(self):
        return {'session': {
            "batch": self.batch,
            "time": self.ann_time,
            "data_type": self.data_type,
            "task_type": self.task_type,
            "question": self.question,
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "response": self.response,
            "vote_label": self.vote_label
        }}

    def __str__(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)


def rm_eval(infile, ansfile):
    """
    评测rm, 计算top1比例/满分比例/pair acc
    args:
        infile: inference file
        ansfile: test file
    """

    for line in read_jsonl(ansfile):
        line['session']['vote_label'].split(',')[0].split('=')