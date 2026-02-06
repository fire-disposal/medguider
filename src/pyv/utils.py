import numpy as np
from typing import List, Tuple
from .models import PatientProfile, EscortProfile
from .config import CONFIG

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
