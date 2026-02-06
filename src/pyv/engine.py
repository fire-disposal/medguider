import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    accuracy_score,
)
from .models import PatientProfile, EscortProfile, MatchScoreDetail, DatasetSchema

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
