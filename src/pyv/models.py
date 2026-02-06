from dataclasses import dataclass
from typing import List, TypedDict

@dataclass(frozen=True)
class PatientProfile:
    """系统输入：患者建模原型"""
    pid: str
    gender: str             # 性别
    age: int                # 年龄
    education: str          # 学历
    survey_tags: List[str]  # 自填量表标签 (e.g., ["高龄独居", "自理困难"])
    orders: List[str]       # 历史订单/服务类目 (e.g., ["诊间全流程陪同"])
    service_frequency: int  # 活跃频率 (次/月)


@dataclass(frozen=True)
class EscortProfile:
    """系统输入：陪诊员建模原型"""
    eid: str
    name: str
    gender: str             # 性别
    age: int                # 年龄
    education: str          # 学历
    certs: List[str]        # 职业证书 (e.g., ["执业护士证"])
    avg_rating: float       # 平均得分 (0.0-5.0)
    keywords: List[str]     # NLP提取标签 (e.g., ["细心", "专业"])
    historical_orders: int  # 历史完成单量


@dataclass(frozen=True)
class MatchScoreDetail:
    """系统输出：评分明细"""
    total_score: float
    ability_component: float
    attitude_component: float
    similarity_component: float
    active_component: float


class DatasetSchema(TypedDict):
    """训练/测试集规格要求"""
    patient_id: str
    escort_id: str
    score: float  # 系统预测分
    truth: int  # 专家标注 (0 or 1)
    pred: int  # 系统判定 (0 or 1)
    # 增加组件分项用于深度分析
    ability: float
    attitude: float
    similarity: float
    active: float
    # 增加人口学维度用于分层分析
    p_age: int
    e_cert_count: int
