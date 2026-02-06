"""
================================================================================
【科研设计与数据规范：老年陪诊推荐系统】
================================================================================

1. 原始数据获取逻辑 (Data Acquisition):
   - 定量途径：通过《门诊老年患者陪诊服务需求量表》获取 300+ 样本数据。
   - 定性途径：对 15-20 名老年患者进行半结构式访谈，提取“成长发展属性”等高维特征。
   - 整合方式：量性研究进行聚类分析（确定画像类别），质性研究进行特征补全与标签凝练。

2. 问卷变量设计要求 (Variable Design):
   - 一级标签：
     * 基础属性 (Base): Age, Education
     * 生存属性 (Life): Disease_Type, ADL_Score (自理能力)
     * 关系属性 (Relation): Living_Status (独居/同住)
     * 成长属性 (Growth): Digital_Literacy (数字素养/自我学习态度)
   - 二级标签（标签化文本）:
     * 例如：“我不会用智能手机挂号” -> 归类为“Digital_Literacy:Low”
     * 例如：“我怕给子女添麻烦” -> 归类为“Relation:Fear_of_Burden”

3. 算法参数要求:
   - Apriori支持度 (min_support): 建议设定在 [0.1, 0.2] 之间，确保护理需求的长尾特征不被过滤。
   - 匹配权重 (mu, rho, lambda): 
     * mu (核心能力) 权重应最高，体现医疗服务的安全性。
     * lambda (相似度) 负责画像的个性化精准推送。
================================================================================
"""

import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# 解决 Matplotlib 中文显示问题 (增加更多备选字体)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False 
sns.set_theme(font='Microsoft YaHei', style='whitegrid') # 使 seaborn 也支持中文，并优化风格

# ==========================================
# 实验模块 A：问卷数据模拟（严格对应申报书标签体系）
# ==========================================

# 模拟 3.2.2.2 中的标签提炼过程：将访谈内容转化为可计算标签
def get_simulated_survey_data():
    """
    模拟问卷反馈数据，包含随机分布以体现真实长尾特征。
    """
    np.random.seed(42)
    scenarios = [
        # 画像一：高龄失能独居型 (核心特征：80+, Alone, 严重受损)
        (['Age:80+', 'ADL:严重受损', 'Live:Alone', 'Disease:慢性病'], ['Srv:诊间陪同', 'Srv:代取药', 'Srv:报告解读'], 0.35),
        # 画像二：积极学习社交型 (核心特征：70-80, 学习态度积极)
        (['Age:70-80', 'Growth:学习态度积极', 'Live:WithKids', 'Edu:大学'], ['Srv:导医引导', 'Srv:心理安抚'], 0.25),
        # 画像三：术后康复困难型 (核心特征：ADL受损, 术后)
        (['Age:70-80', 'ADL:受损', 'Disease:术后', 'Relation:怕麻烦子女'], ['Srv:术后陪送', 'Srv:预约挂号', 'Srv:代取药'], 0.20),
        # 画像四：数字鸿沟型 (核心特征：Digital:Low, 焦虑)
        (['Age:60-70', 'Digital:Low', 'Emotion:焦虑', 'Disease:高血压'], ['Srv:预约挂号', 'Srv:导医引导', 'Srv:报告解读'], 0.20)
    ]
    
    data = []
    for _ in range(350):
        # 按权重分配画像
        r = np.random.random()
        cumulative = 0
        selected_scenario = scenarios[0]
        for tags, srvs, weight in scenarios:
            cumulative += weight
            if r <= cumulative:
                selected_scenario = (tags, srvs)
                break
        
        tags, srvs = selected_scenario
        # 模拟组合特征
        record = tags + srvs
        # 随机添加 0-1 个随机背景标签
        if np.random.random() > 0.7:
             record.append(np.random.choice(['Economic:良好', 'Economic:一般', 'Live:养老院']))
             
        data.append(record)
        
    return data

# ==========================================
# 实验模块 B：3.2.3 算法核心框架
# ==========================================

class ResearchEngine:
    def __init__(self, data):
        self.raw_data = data
        self.rules = None
        self._prepare_data()

    def _prepare_data(self):
        """对应 3.2.2.2 画像可视化预处理"""
        te = TransactionEncoder()
        te_ary = te.fit(self.raw_data).transform(self.raw_data)
        # 显式将列名转换为标准 Python 字符串类型，防止 mlxtend 在 NumPy 2.0 环境下
        # 误将 np.str_ 识别为 np.generic 并尝试进行 int() 转换
        self.df = pd.DataFrame(te_ary, columns=[str(c) for c in te.columns_])

    def run_apriori(self, min_support=0.15):
        """
        实现 3.2.3(2) 的 Apriori 学习模型。
        通过频繁项集挖掘，发现画像(antecedents)与需求(consequents)之间的关联。
        """
        frequent_itemsets = apriori(self.df, min_support=min_support, use_colnames=True)
        
        # 针对 NumPy 2.0+ 与 mlxtend 的兼容性补丁：
        # 确保 itemsets 中的项是标准 Python 字符串，而不是 np.str_
        if not frequent_itemsets.empty:
            frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(
                lambda x: frozenset(str(item) for item in x)
            )

        # 计算提升度(Lift)，寻找比随机推荐效果更好的关联规则
        self.rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
        return self.rules

    def calculate_u_score(self, patient_tags, escort_profile, weights={'mu': 0.5, 'rho': 0.3, 'lambda': 0.2}):
        """
        实现申报书公式 (1) 和 (3) 的契合评分 U。
        U = Wk * Sim(euk, jk) + M_factor
        """
        # 1. 资质分 (对应公式1中的 L(Ju) 和 A(Ju))
        # 核心能力 (ability) 权重最高，态度 (attitude) 为辅
        m_factor = weights['mu'] * escort_profile.get('ability', 0.5) + weights['rho'] * escort_profile.get('attitude', 0.5)
        
        # 2. 相似度 Sim (对应公式3：患者特征与陪诊员专长的契合度)
        patient_needs = [t for t in patient_tags if t.startswith('Srv:')]
        if not patient_needs: # 如果没明确服务需求，则看所有标签的匹配度
            patient_needs = patient_tags
            
        matches = len(set(patient_needs) & set(escort_profile.get('specialties', [])))
        sim = matches / len(patient_needs) if patient_needs else 0
        
        # 3. 最终加权评分
        u_score = (weights['lambda'] * sim) + m_factor
        return round(u_score, 4)

    def recommend_best_escorts(self, patient_tags, candidates, top_n=3):
        """更贴近现实的多候选人排名逻辑"""
        results = []
        for cand in candidates:
            score = self.calculate_u_score(patient_tags, cand)
            results.append({**cand, 'match_score': score})
        
        # 按得分从高到低排序
        ranked = sorted(results, key=lambda x: x['match_score'], reverse=True)
        return ranked[:top_n]

# ==========================================
# 实验执行与呈现
# ==========================================

# 1. 初始化引擎
lab = ResearchEngine(get_simulated_survey_data())

# 2. 算法挖掘（发现画像与需求关联）
found_rules = lab.run_apriori()

# 3. 多场景模拟推荐 (更贴近现实)
test_patient = ['Age:80+', 'Live:Alone', 'ADL:严重受损', 'Srv:诊间陪同', 'Srv:代取药']

escort_pool = [
    {'name': '张姐 (资深医护)', 'ability': 0.98, 'attitude': 0.85, 'specialties': ['诊间陪同', '代取药', '慢性病护理', '术后陪送']},
    {'name': '李哥 (暖心向导)', 'ability': 0.75, 'attitude': 0.98, 'specialties': ['导医引导', '诊间陪同', '心理安抚']},
    {'name': '陈老师 (专业代办)', 'ability': 0.90, 'attitude': 0.80, 'specialties': ['预约挂号', '报告解读', '代取药']},
    {'name': '小王 (新晋陪诊)', 'ability': 0.65, 'attitude': 0.90, 'specialties': ['导医引导', '诊间陪同']},
]

top_recommendations = lab.recommend_best_escorts(test_patient, escort_pool)

print("-" * 60)
print("【老年陪诊多因子推荐实验】")
print(f"  > 目标患者特征: {test_patient}")
print(f"  > 系统发现规则: {len(found_rules)} 条")
print("  > 推荐优先级排名:")
for i, rec in enumerate(top_recommendations):
    stars = "★" * int(rec['match_score'] * 10)
    print(f"    [{i+1}] {rec['name']} | 评分: {rec['match_score']} | 专长: {rec['specialties']} {stars}")
print("-" * 60)

# ==========================================
# 3.2.2.2 & 3.2.3 可视化增强 (2x2 专业看板)
# ==========================================

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# 1. 高频特征标签分布 (显式转换索引解决 ValueError)
self_tags_count = lab.df.sum().sort_values(ascending=True).tail(12)
y_labels = [str(x) for x in self_tags_count.index]
colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(self_tags_count)))
bars = axes[0, 0].barh(y_labels, self_tags_count.values, color=colors, edgecolor='gray', alpha=0.8)
axes[0, 0].set_title('① 老年患者高频特征标签分布 (画像底稿)', fontsize=13, fontweight='bold')
axes[0, 0].grid(axis='x', linestyle='--', alpha=0.5)

# 2. 核心标签「特征-需求」关联强度热图 (非对称映射)
trait_tags = [t for t in lab.df.columns if not t.startswith('Srv:')]
srv_tags = [t for t in lab.df.columns if t.startswith('Srv:')]

# 选取 Top 10 患者特征与 Top 8 核心服务需求
top_traits = lab.df[trait_tags].sum().sort_values(ascending=False).head(10).index
top_srvs = lab.df[srv_tags].sum().sort_values(ascending=False).head(8).index

# 构建交叉关联概率矩阵: P(服务|特征) = (特征&服务同时出现) / (特征出现总计)
relation_matrix = pd.DataFrame(index=top_traits, columns=top_srvs)
for t in top_traits:
    for s in top_srvs:
        joint_count = ((lab.df[t] == 1) & (lab.df[s] == 1)).sum()
        trait_count = (lab.df[t] == 1).sum()
        relation_matrix.loc[t, s] = joint_count / trait_count if trait_count > 0 else 0

relation_matrix = relation_matrix.astype(float)
sns.heatmap(relation_matrix, annot=True, fmt=".2f", cmap='YlGnBu', ax=axes[0, 1], 
            cbar_kws={'label': '关联概率 P(服务|特征)'})
axes[0, 1].set_title('② 患者特征 -> 服务需求关联强度图', fontsize=13, fontweight='bold')
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')

# 3. 推荐算法多因子评分排名 (真实模拟可视化)
names = [r['name'] for r in reversed(top_recommendations)]
match_scores = [r['match_score'] for r in reversed(top_recommendations)]
colors_res = plt.cm.summer(np.linspace(0.3, 0.7, len(names)))
axes[1, 0].barh(names, match_scores, color=colors_res, edgecolor='gray')
axes[1, 0].set_xlim(0, 1.1)
axes[1, 0].set_title(f'③ 目标患者实时匹配策略排名\n(特征: {", ".join(test_patient[:3])}...)', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('契合度评分 (Score U)')
for i, v in enumerate(match_scores):
    axes[1, 0].text(v + 0.02, i, f"{v:.4f}", va='center', fontweight='bold')

# 4. 需求关联网络图 (显式转换节点类型)
if not found_rules.empty:
    G = nx.DiGraph()
    top_rules = found_rules.sort_values('lift', ascending=False).head(20)
    for _, row in top_rules.iterrows():
        ant = str(list(row['antecedents'])[0])
        con = str(list(row['consequents'])[0])
        G.add_edge(ant, con, weight=float(row['lift']))
    
    pos = nx.spring_layout(G, k=0.7, seed=42)
    nx.draw_networkx_nodes(G, pos, ax=axes[1, 1], node_size=1000, node_color='orange', alpha=0.7)
    nx.draw_networkx_edges(G, pos, ax=axes[1, 1], edge_color='gray', arrows=True, width=1.2, alpha=0.4)
    nx.draw_networkx_labels(G, pos, ax=axes[1, 1], font_size=9, font_family='Microsoft YaHei')
    axes[1, 1].set_title('④ 需求-特征挖掘知识图谱', fontsize=13, fontweight='bold')
    axes[1, 1].axis('off')

plt.suptitle('老年陪诊服务「需求挖掘-匹配推荐」全链路仿真实验看板', fontsize=18, fontweight='bold', y=0.96)

try:
    print("\n[提示] 可视化看板已生成，请在弹出窗口查看。")
    print("[提示] 若需退出，请直接关闭图片窗口，避免使用 Ctrl+C。")
    # 同时保存一份到本地，防止 GUI 无法显示
    plt.savefig('research_dashboard.png', dpi=300, bbox_inches='tight')
    print("[完成] 实验图表已同步保存至: research_dashboard.png")
    plt.show()
except KeyboardInterrupt:
    print("\n[信息] 用户中断了程序进度。")
finally:
    plt.close('all') # 释放内存