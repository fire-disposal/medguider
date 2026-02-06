"""
================================================================================
【科研数据预处理：老年陪诊需求问卷清洗脚本】
================================================================================
功能：
1. 原始问卷数据模拟与加载
2. 连续变量离散化（Age -> AgeRange）
3. 量表得分计算与等级划分（ADL分级）
4. 语义标签化整合（生成适用于 Apriori 的事务数据）
================================================================================
"""

import pandas as pd
import numpy as np


class SurveyCleaner:
    def __init__(self):
        # 定义全局标签映射标准，确保与推荐引擎一致
        self.label_prefix = {
            "age": "Age:",
            "edu": "Edu:",
            "live": "Live:",
            "adl": "ADL:",
            "digital": "Digital:",
            "disease": "Disease:",
            "srv": "Srv:",
        }

    def simulate_raw_data(self, n=100):
        """模拟原始问卷导出数据（Excel/CSV格式）"""
        np.random.seed(42)
        data = {
            "用户ID": range(1, n + 1),
            "年龄": np.random.randint(60, 95, n),
            "文化程度": np.random.choice(
                ["小学及以下", "初中", "高中/中专", "大学及以上"], n
            ),
            "居住情况": np.random.choice(["独居", "与子女同住", "养老院", "其他"], n),
            "ADL原始分": np.random.randint(
                40, 100, n
            ),  # 假设100分制，分数越低自理能力越差
            "疾病类型": np.random.choice(
                ["慢性病", "术后恢复", "急性病", "健康体检"], n
            ),
            "数字素养反馈": np.random.choice(
                ["完全不会用手机", "只会打接电话", "会用微信", "能熟练挂号支付"], n
            ),
            # 模拟多选题转化后的 0/1 列
            "需求_诊间陪同": np.random.choice([0, 1], n, p=[0.6, 0.4]),
            "需求_代取药": np.random.choice([0, 1], n, p=[0.7, 0.3]),
            "需求_报告解读": np.random.choice([0, 1], n, p=[0.5, 0.5]),
            "需求_预约挂号": np.random.choice([0, 1], n, p=[0.8, 0.2]),
        }
        return pd.DataFrame(data)

    def clean_process(self, df):
        """执行标准化模型清洗流程"""
        records = []

        for idx, row in df.iterrows():
            current_tags = []

            # 1. 年龄离散化
            age = row["年龄"]
            if age >= 80:
                tag = "80+"
            elif age >= 70:
                tag = "70-80"
            else:
                tag = "60-70"
            current_tags.append(self.label_prefix["age"] + tag)

            # 2. ADL分级逻辑
            adl_score = row["ADL原始分"]
            if adl_score <= 60:
                adl_tag = "严重受损"
            elif adl_score <= 85:
                adl_tag = "受损"
            else:
                adl_tag = "自理良好"
            current_tags.append(self.label_prefix["adl"] + adl_tag)

            # 3. 语义映射：居住情况
            live_map = {
                "独居": "Alone",
                "与子女同住": "WithKids",
                "养老院": "NursingHome",
            }
            current_tags.append(
                self.label_prefix["live"] + live_map.get(row["居住情况"], "Other")
            )

            # 4. 数字素养简化
            digital_val = row["数字素养反馈"]
            if digital_val in ["完全不会用手机", "只会打接电话"]:
                d_tag = "Low"
            else:
                d_tag = "High"
            current_tags.append(self.label_prefix["digital"] + d_tag)

            # 5. 疾病标签直接转化
            current_tags.append(self.label_prefix["disease"] + row["疾病类型"])

            # 6. 提取服务需求 (多选题列)
            srv_cols = [c for c in df.columns if c.startswith("需求_")]
            for col in srv_cols:
                if row[col] == 1:
                    srv_name = col.replace("需求_", "")
                    current_tags.append(self.label_prefix["srv"] + srv_name)

            records.append(current_tags)

        return records


# ==========================================
# 脚本执行示例
# ==========================================

if __name__ == "__main__":
    cleaner = SurveyCleaner()

    # 步骤一：读取/模拟数据
    raw_df = cleaner.simulate_raw_data(10)  # 模拟10条示例
    print("--- 原始数据样本 (前3条) ---")
    print(raw_df[["年龄", "ADL原始分", "居住情况", "数字素养反馈"]].head(3))

    # 步骤二：清洗与标签化
    cleaned_records = cleaner.clean_process(raw_df)

    print("\n--- 清洗后的事务数据样本 (Ready for Apriori) ---")
    for i in range(3):
        print(f"ID {i+1}: {cleaned_records[i]}")

    # 步骤三：说明导出
    print("\n[提示] 该列表结构已可直接作为 ResearchEngine 的输入。")
    print("[提示] 建议在实际使用时通过 pd.read_excel('survey.xlsx') 替换模拟函数。")
