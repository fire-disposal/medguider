"""
================================================================================
【科研设计与工程落地：老年陪诊推荐系统 v0.2】
================================================================================
更新说明：
1. 配置化管理：引入 CONFIG 字典集中控制所有模型权重与参数。
2. 增强建模：
   - 患者侧：集成自填量表、历史订单类目、服务频率。
   - 陪诊侧：集成职业证书、用户评分、NLP提取关键词。
3. 训练与评价：引入模拟“专家标准”的训练集与测试集，量化匹配效果。
4. 闭环优化：模拟评价反馈对模型精度的持续提升。
================================================================================
"""

import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional, TypedDict
from dataclasses import dataclass, asdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    confusion_matrix
)
import matplotlib.pyplot as plt

# ==========================================
# 0. 强类型定义 (Strongly Typed Definitions)
# ==========================================


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
    historical_orders: int  # 新增：历史完成单量


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


# ==========================================
# 1. 配置中心 (Centralized Configuration)
# ==========================================
CONFIG = {
    "weights": {
        "ability_mu": 0.45,  # 职业技能/证书加权得分占比
        "attitude_rho": 0.15,  # 用户评分得分占比
        "similarity_lambda": 0.30,  # 画像匹配度 (Tags vs Keywords/Certs) 占比
        "active_iota": 0.10,  # 服务频率/经验活跃度占比
    },
    "params": {
        "test_size": 0.2,
        "random_seed": 42,
        "satisfaction_threshold": 0.55,  # 判定为“推荐”的最低综合分
        "scaling": {
            "max_ability_sum": 1.5,  # 达到满分能力的证书权重累加值
            "min_valid_rating": 3.0,  # 评分映射的基准值
            "max_valid_rating": 5.0,  # 评分映射的最高值
            "max_active_freq": 30,  # 达到满分活跃度的月服务频率
        },
        "expert_standard": {
            "core_certs": ["执业护士证", "高级护理员"],
            "min_rating": 4.0,
        },
    },
    # 集中化标签系统定义
    "labels": {
        "demographics": {
            "genders": ["男", "女"],
            "education": ["小学及以下", "初中", "高中/中专", "专科/本科", "研究生及以上"]
        },
        "patient": {
            "survey": ["高龄独居", "数字鸿沟", "自理困难", "慢病随访", "术后康复"],
            "orders": [
                "系统预约挂号",
                "诊间全流程陪同",
                "诊后代取药/寄送",
                "心理安抚辅导",
            ],
        },
        "escort": {
            "certs": {
                "执业护士证": 1.0,
                "高级护理员": 0.8,
                "心理咨询师": 0.7,
                "红十字急救证": 0.6,
                "健康管理师": 0.5,
            },
            "keywords": ["细心", "沟通力强", "体力好", "专业规范", "温柔耐心"],
        },
    },
    # 业务逻辑映射：患者需求 -> 陪诊员特征/证书 (用于计算 Similarity)
    "match_logic": {
        "高龄独居": ["细心", "沟通力强", "心理咨询师"],
        "自理困难": ["体力好", "执业护士证", "高级护理员"],
        "术后康复": ["执业护士证", "专业规范", "红十字急救证"],
        "心理安抚辅导": ["心理咨询师", "温柔耐心", "沟通力强"],
        "诊间全流程陪同": ["专业规范", "沟通力强", "体力好"],
    },
    # 业务逻辑映射：患者需求 -> 陪诊员特征/证书 (用于计算 Similarity)
    "semantic_space": {
        "高龄独居": {"细心": 0.9, "心理咨询师": 0.7, "沟通力强": 0.8},
        "自理困难": {"体力好": 0.9, "高级护理员": 1.0, "执业护士证": 0.8},
        "术后康复": {"执业护士证": 1.0, "专业规范": 0.8, "红十字急救证": 0.7},
        "心理安抚辅导": {"心理咨询师": 1.0, "温柔耐心": 0.9, "沟通力强": 0.8},
        "诊间全流程陪同": {"专业规范": 1.0, "沟通力强": 0.8, "体力好": 0.7},
        "系统预约挂号": {"专业规范": 0.9, "沟通力强": 0.6},
        "诊后代取药/寄送": {"细心": 0.8, "专业规范": 0.7},
    },
}

# 解决 Matplotlib 中文显示问题
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimSun", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(font="Microsoft YaHei", style="whitegrid")

# ==========================================
# 2. 实验模块：数据模拟与“专家标准”生成
# ==========================================


def generate_engineering_data(
    n: int = 500,
) -> Tuple[List[PatientProfile], List[EscortProfile]]:
    """
    增强版模拟数据生成器：
    引入“人群画像”原型，生成更具业务特征分布的样本，包含人口学变量。
    """
    np.random.seed(CONFIG["params"]["random_seed"])
    patients: List[PatientProfile] = []
    escorts: List[EscortProfile] = []

    demos = CONFIG["labels"]["demographics"]
    p_survey_pool = CONFIG["labels"]["patient"]["survey"]
    p_order_pool = CONFIG["labels"]["patient"]["orders"]
    e_cert_pool = list(CONFIG["labels"]["escort"]["certs"].keys())
    e_kw_pool = CONFIG["labels"]["escort"]["keywords"]

    # 1. 生成陪诊员分布
    for i in range(n):
        # 模拟人口学变量
        gender = np.random.choice(demos["genders"], p=[0.3, 0.7]) # 陪诊员女性比例通常偏高
        age = np.random.randint(25, 55)
        edu = np.random.choice(demos["education"], p=[0.05, 0.15, 0.4, 0.35, 0.05])

        # 模拟三种画像：资深型 (20%)、标准型 (60%)、新手型 (20%)
        rand_val = np.random.random()
        if rand_val < 0.2:  # 资深型
            certs = np.random.choice(e_cert_pool, size=3, replace=False).tolist()
            rating = np.random.uniform(4.5, 5.0)
            kws = np.random.choice(e_kw_pool, size=3, replace=False).tolist()
            orders = np.random.randint(50, 200)
        elif rand_val < 0.8:  # 标准型
            certs = np.random.choice(
                e_cert_pool, size=np.random.randint(1, 3), replace=False
            ).tolist()
            rating = np.random.uniform(3.5, 4.5)
            kws = np.random.choice(e_kw_pool, size=2, replace=False).tolist()
            orders = np.random.randint(10, 50)
        else:  # 新手型
            # 模拟可能未持有证书的情况 (50% 概率一张证都没有)
            if np.random.random() < 0.5:
                certs = []
            else:
                certs = [np.random.choice(e_cert_pool)]
            rating = np.random.uniform(3.0, 4.0)
            kws = np.random.choice(e_kw_pool, size=1, replace=False).tolist()
            orders = np.random.randint(0, 10)

        e = EscortProfile(
            eid=f"E{i:03d}",
            name=f"陪诊员_{i}",
            gender=gender,
            age=age,
            education=edu,
            certs=certs,
            avg_rating=round(rating, 2),
            keywords=kws,
            historical_orders=orders,
        )
        escorts.append(e)

    # 2. 生成患者分布
    for i in range(n):
        # 模拟人口学变量
        p_gender = np.random.choice(demos["genders"])
        p_age = np.random.randint(60, 95)
        p_edu = np.random.choice(demos["education"], p=[0.4, 0.3, 0.2, 0.08, 0.02])

        # 模拟：需求频率分布（长尾或正态）
        freq = int(np.random.exponential(scale=10) % 30) + 1
        p = PatientProfile(
            pid=f"P{i:03d}",
            gender=p_gender,
            age=p_age,
            education=p_edu,
            survey_tags=np.random.choice(
                p_survey_pool, size=np.random.randint(1, 4), replace=False
            ).tolist(),
            orders=np.random.choice(p_order_pool, size=1, replace=False).tolist(),
            service_frequency=freq,
        )
        patients.append(p)

    return patients, escorts


# ==========================================
# 3. 核心推荐算法 (V0.2 Engineering Engine)
# ==========================================


class EngineeringEngine:
    def __init__(self, config):
        self.config = config
        self.weights = config["weights"]

    def calculate_similarity_advanced(self, patient: PatientProfile, escort: EscortProfile) -> float:
        """改进后的语义加权匹配算法"""
        total_needs = patient.survey_tags + patient.orders
        if not total_needs:
            return 0.0

        match_scores = []
        escort_features = set(escort.keywords) | set(escort.certs)

        for need in total_needs:
            # 获取该需求在语义空间中关联的所有特征
            related_features = self.config["semantic_space"].get(need, {})
            # 计算交集部分的权重得分
            score = sum(related_features[f] for f in related_features if f in escort_features)
            # 归一化：单项需求匹配最高分为 1.0
            match_scores.append(min(score, 1.0))

        return sum(match_scores) / len(total_needs)

    def calculate_match_score(
        self, patient: PatientProfile, escort: EscortProfile
    ) -> MatchScoreDetail:
        """核心匹配算法"""
        scaling = self.config["params"]["scaling"]

        # 1. Ability Score (归一化)
        cert_weights = self.config["labels"]["escort"]["certs"]
        raw_ability = sum(cert_weights.get(c, 0) for c in escort.certs)
        ability_score = min(raw_ability / scaling["max_ability_sum"], 1.0)

        # 2. Attitude Score (0-1)
        r_min, r_max = scaling["min_valid_rating"], scaling["max_valid_rating"]
        attitude_score = (escort.avg_rating - r_min) / (r_max - r_min)

        # 3. Similarity Score (V0.3 Advanced)
        similarity_score = self.calculate_similarity_advanced(patient, escort)

        # 4. Experience/Active Score (V0.3 Corrected to Escort Experience)
        active_score = min(escort.historical_orders / scaling["max_active_freq"], 1.0)

        # 加权计算
        u = (
            self.weights["ability_mu"] * ability_score
            + self.weights["attitude_rho"] * attitude_score
            + self.weights["similarity_lambda"] * similarity_score
            + self.weights["active_iota"] * active_score
        )

        return MatchScoreDetail(
            total_score=round(u, 4),
            ability_component=round(ability_score, 4),
            attitude_component=round(attitude_score, 4),
            similarity_component=round(similarity_score, 4),
            active_component=round(active_score, 4),
        )

    def evaluate_system(
        self, patients: List[PatientProfile], escorts: List[EscortProfile]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
        """
        引入专家标准进行离线评估。
        """
        expert_cfg = self.config["params"]["expert_standard"]
        dataset: List[DatasetSchema] = []

        for p, e in zip(patients, escorts):
            detail = self.calculate_match_score(p, e)
            score = detail.total_score

            # 专家标准逻辑 (消除硬编码字符串)
            has_hard_cert = any(c in e.certs for c in expert_cfg["core_certs"])
            good_rating = e.avg_rating > expert_cfg["min_rating"]
            core_met = any(
                set(self.config["match_logic"].get(n, []))
                & (set(e.keywords) | set(e.certs))
                for n in (p.survey_tags + p.orders)
            )

            expert_label = 1 if (has_hard_cert and good_rating and core_met) else 0
            prediction = (
                1 if score >= self.config["params"]["satisfaction_threshold"] else 0
            )

            dataset.append(
                {
                    "patient_id": p.pid,
                    "escort_id": e.eid,
                    "score": score,
                    "truth": expert_label,
                    "pred": prediction,
                    "ability": detail.ability_component,
                    "attitude": detail.attitude_component,
                    "similarity": detail.similarity_component,
                    "active": detail.active_component,
                    "p_age": p.age,
                    "e_cert_count": len(e.certs)
                }
            )

        df = pd.DataFrame(dataset)
        train, test = train_test_split(
            df, test_size=self.config["params"]["test_size"], random_state=42
        )

        accuracy = accuracy_score(test["truth"], test["pred"])
        precision = precision_score(test["truth"], test["pred"], zero_division=0)

        return train, test, accuracy, precision


# ==========================================
# 4. 执行与可视化
# ==========================================


def run_simulation_v03():
    # ANSI 颜色定义
    class Color:
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        BLUE = "\033[94m"
        BOLD = "\033[1m"
        END = "\033[0m"

    # 数据准备
    patients, escorts = generate_engineering_data(600)
    engine = EngineeringEngine(CONFIG)

    # 【优化】预先建立查找字典：O(1) 复杂度
    patient_map = {p.pid: p for p in patients}
    escort_map = {e.eid: e for e in escorts}

    # 模拟实验
    train_df, test_df, acc, prec = engine.evaluate_system(patients, escorts)
    rec = recall_score(test_df["truth"], test_df["pred"], zero_division=0)
    f1 = f1_score(test_df["truth"], test_df["pred"], zero_division=0)

    print("-" * 60)
    print(f"{Color.BOLD}【工程化推荐系统 v0.3 测试报告摘要】{Color.END}")
    print(f" > 当前配置权重: {CONFIG['weights']}")
    print(f" > 决策阈值 (Threshold): {CONFIG['params']['satisfaction_threshold']}")
    print(" > 测试指标: ")
    print(f"   - 准确率 (Acc): {acc:.4f}")
    print(f"   - 精确率 (Pre): {prec:.4f}")
    print(f"   - 召回率 (Rec): {rec:.4f}")
    print(f"   - F1 分数: {f1:.4f}")
    print("-" * 60)

    # 深度日志输出：展示 Top 5 最佳搭配
    print(f"\n{Color.BOLD}>>> 最佳匹配案例深度解析 (TOP 5):{Color.END}")
    top_matches = test_df.sort_values(by="score", ascending=False).head(5)

    for _, row in top_matches.iterrows():
        p_obj = patient_map[row["patient_id"]]
        e_obj = escort_map[row["escort_id"]]

        # 根据分值选择颜色
        score = row["score"]
        if score >= 0.75:
            color = Color.GREEN
        elif score >= 0.55:
            color = Color.YELLOW
        else:
            color = Color.RED

        print(
            f"{color}[匹配分: {score:.4f}]{Color.END} 患者: {p_obj.pid} [{p_obj.gender}/{p_obj.age}岁/{p_obj.education}] ({', '.join(p_obj.survey_tags)})"
        )
        print(
            f"        └─ 推荐给: {e_obj.name} [{e_obj.gender}/{e_obj.age}岁/{e_obj.education}] (评分: {e_obj.avg_rating}, 接单: {e_obj.historical_orders}, 证书: {', '.join(e_obj.certs) if e_obj.certs else '无'})"
        )
        # 输出评分明细
        detail = engine.calculate_match_score(p_obj, e_obj)
        print(
            f"        {Color.BLUE}分析: 能力[{detail.ability_component}] 语义契合[{detail.similarity_component}] 活跃/经验[{detail.active_component}]{Color.END}"
        )

    print("-" * 60)

    # ==========================================
    # 可视化看板 (工程测试 v3.0：语义空间与活跃度改进)
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(22, 14))
    plt.subplots_adjust(wspace=0.25, hspace=0.3)

    # 1. Precision-Recall 曲线 (左上)
    precisions_curve, recalls_curve, _ = precision_recall_curve(test_df["truth"], test_df["score"])
    axes[0, 0].plot(recalls_curve, precisions_curve, label="P-R Curve", color="darkgreen", linewidth=2)
    axes[0, 0].set_xlabel("召回率 (Recall)")
    axes[0, 0].set_ylabel("精确率 (Precision)")
    axes[0, 0].set_title("① 系统核心：P-R 性能曲线与当前运行点", fontsize=14, fontweight="bold")
    axes[0, 0].plot(rec, prec, "ro", label=f"当前点 (T={CONFIG['params']['satisfaction_threshold']})")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 组件相关性热图 (右上 - 观察哪些维度主导了系统决策)
    component_cols = ["ability", "attitude", "similarity", "active", "score", "truth"]
    corr_matrix = test_df[component_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="vlag", center=0, ax=axes[0, 1])
    axes[0, 1].set_title("② 决策因子热图：分项评分与最终结果的相关性", fontsize=14, fontweight="bold")

    # 3. 人群分层热图 (左下 - 观察不同年龄/资质下的系统适配度)
    # 将年龄分桶
    test_df['age_group'] = pd.cut(test_df['p_age'], bins=[60, 70, 80, 95], labels=['60-70岁', '70-80岁', '80+岁'])
    # 计算不同交叉组的平均分
    pivot_table = test_df.pivot_table(values='score', index='age_group', columns='e_cert_count', aggfunc='mean')
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[1, 0])
    axes[1, 0].set_title("③ 人群分层热图：不同年龄组 vs 陪诊资质的平均分分布", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("陪诊员持有证书数量")
    axes[1, 0].set_ylabel("患者年龄组")

    # 4. 误差分布矩阵 (右下 - 识别“误报”与“漏报”的重灾区)
    cm = confusion_matrix(test_df["truth"], test_df["pred"])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", ax=axes[1, 1],
                xticklabels=["不推荐", "推荐"], yticklabels=["专家拒荐", "专家首选"])
    axes[1, 1].set_title("④ 误差深度矩阵：系统决策与专家标准的一致性分析", fontsize=14, fontweight="bold")

    plt.suptitle("老年陪诊推荐系统 v0.3 工程测试与数据核心分析面板", fontsize=20, fontweight="bold", y=0.96)
    plt.savefig("engineering_v03_core_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    run_simulation_v03()
