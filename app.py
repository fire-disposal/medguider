import os
import sys

try:
    from src.medguider import CONFIG, EngineeringEngine, generate_engineering_data, PatientProfile
except Exception:
    # 兼容直接在项目根目录运行或被其他 runner 导入的情况：确保可导入 `medguider`
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from medguider import CONFIG, EngineeringEngine, generate_engineering_data, PatientProfile


def main():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import json
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import recall_score, f1_score, precision_recall_curve, confusion_matrix

    # 页面配置（必须在其他 st.* 之前调用）
    st.set_page_config(page_title="老年陪诊推荐系统 v0.3", layout="wide")

    # 初始化 Session State
    if "config" not in st.session_state:
        st.session_state.config = CONFIG.copy()

    # 侧边栏：配置参数
    st.sidebar.header("⚙️ 系统参数配置")

    with st.sidebar.expander("权重分分配 (Weights)", expanded=True):
        st.session_state.config["weights"]["ability_mu"] = st.slider(
            "职业能力权重 (Ability)", 0.0, 1.0, st.session_state.config["weights"]["ability_mu"]
        )
        st.session_state.config["weights"]["attitude_rho"] = st.slider(
            "服务态度权重 (Attitude)", 0.0, 1.0, st.session_state.config["weights"]["attitude_rho"]
        )
        st.session_state.config["weights"]["similarity_lambda"] = st.slider(
            "画像契合权重 (Similarity)", 0.0, 1.0, st.session_state.config["weights"]["similarity_lambda"]
        )
        st.session_state.config["weights"]["active_iota"] = st.slider(
            "活跃/经验权重 (Active)", 0.0, 1.0, st.session_state.config["weights"]["active_iota"]
        )

    st.sidebar.markdown("---")
    st.sidebar.header("🧪 实验设置")
    n_samples = st.sidebar.number_input("模拟样本量", 100, 2000, 600)
    threshold = st.sidebar.slider("推荐判定阈值", 0.0, 1.0, st.session_state.config["params"]["satisfaction_threshold"])
    st.session_state.config["params"]["satisfaction_threshold"] = threshold

    # 主界面
    st.title("🏥 老年陪诊推荐系统 - 交互化实验平台")
    st.markdown(
        """
    本系统通过**多维度画像建模**（能力、态度、需求契合度、活跃度）为老年患者推荐最合适的陪诊员。
    您可以通过左侧边栏调整算法权重，实时观察系统在专家标准下的性能表现。
    """
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 匹配实验与评估",
        "🧐 个案深度分析",
        "📊 统计看板",
        "📝 文本统计看板",
        "⚙️ 配置参数"
    ]) 

    # 数据准备
    @st.cache_data
    def get_data(n):
        return generate_engineering_data(n)

    patients, escorts = get_data(n_samples)
    engine = EngineeringEngine(st.session_state.config)

    # 执行评估
    train_df, test_df, acc, prec = engine.evaluate_system(patients, escorts)
    rec = recall_score(test_df["truth"], test_df["pred"], zero_division=0)
    f1 = f1_score(test_df["truth"], test_df["pred"], zero_division=0)

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("准确率 (Acc)", f"{acc:.4f}")
        col2.metric("精确率 (Pre)", f"{prec:.4f}")
        col3.metric("召回率 (Rec)", f"{rec:.4f}")
        col4.metric("F1 分数", f"{f1:.4f}")

        st.subheader("🔝 Top 10 最佳匹配案例")
        patient_map = {p.pid: p for p in patients}
        escort_map = {e.eid: e for e in escorts}
        
        top_matches = test_df.sort_values(by="score", ascending=False).head(10)
        
        match_display = []
        for _, row in top_matches.iterrows():
            p = patient_map[row["patient_id"]]
            e = escort_map[row["escort_id"]]
            match_display.append({
                "匹配分": row["score"],
                "患者ID": p.pid,
                "患者需求": ", ".join(p.survey_tags),
                "陪诊员": e.name,
                "能力分": row["ability"],
                "契合分": row["similarity"],
                "专家结论": "首选" if row["truth"] == 1 else "备选"
            })
        st.table(pd.DataFrame(match_display))

    with tab2:
        st.subheader("🔍 模拟单个患者匹配")
        p_id = st.selectbox("选择测试患者 ID", [p.pid for p in patients[:50]])
        target_p = patient_map[p_id]
        
        st.write(f"**患者画像:** {target_p.gender} | {target_p.age}岁 | {target_p.education}")
        st.write(f"**核心需求:** {', '.join(target_p.survey_tags)}")
        
        st.session_state.target_p = target_p # 为了触发刷新
        
        # 计算当前患者与所有陪诊员的匹配
        scores = []
        for e in escorts:
            detail = engine.calculate_match_score(target_p, e)
            scores.append({
                "eid": e.eid,
                "name": e.name,
                "score": detail.total_score,
                "ability": detail.ability_component,
                "similarity": detail.similarity_component,
                "attitude": detail.attitude_component,
                "active": detail.active_component
            })
        
        scores_df = pd.DataFrame(scores).sort_values(by="score", ascending=False).head(5)
        
        st.write("---")
        st.write("#### 为该患者推荐的 Top 5 陪诊员：")
        for _, s in scores_df.iterrows():
            e_obj = escort_map[s["eid"]]
            col_m1, col_m2 = st.columns([1, 2])
            with col_m1:
                st.info(f"**{s['name']}** (得分: {s['score']:.4f})")
            with col_m2:
                st.write(f"证书: {', '.join(e_obj.certs) if e_obj.certs else '无'} | 评分: {e_obj.avg_rating}")
                # 确保传入 st.progress 的值在 [0.0, 1.0] 范围内，避免 Streamlit 抛出异常
                try:
                    prog_val = float(s.get("score", 0.0))
                except Exception:
                    prog_val = 0.0
                prog_val = max(0.0, min(1.0, prog_val))
                st.progress(prog_val, text=f"综合契合度: {s['score']:.2f}")

    with tab3:
        st.subheader("📈 核心性能可视化")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimSun", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False
        
        # 1. P-R Curve
        precisions_curve, recalls_curve, _ = precision_recall_curve(test_df["truth"], test_df["score"])
        axes[0, 0].plot(recalls_curve, precisions_curve, label="P-R Curve", color="darkgreen")
        axes[0, 0].set_xlabel("Recall")
        axes[0, 0].set_ylabel("Precision")
        axes[0, 0].set_title("P-R Curve")
        axes[0, 0].legend()

        # 2. Correlation
        comp_cols = ["ability", "attitude", "similarity", "active", "score"]
        sns.heatmap(test_df[comp_cols].corr(), annot=True, cmap="vlag", ax=axes[0, 1])
        axes[0, 1].set_title("Feature Correlation")

        # 3. Age Groups
        test_df['age_group'] = pd.cut(test_df['p_age'], bins=[60, 70, 80, 95], labels=['60-70', '70-80', '80+'])
        pivot = test_df.pivot_table(values='score', index='age_group', columns='e_cert_count', aggfunc='mean')
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[1, 0])
        axes[1, 0].set_title("Score by Age & Certs")

        # 4. Confusion Matrix
        cm = confusion_matrix(test_df["truth"], test_df["pred"])
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", ax=axes[1, 1],
                    xticklabels=["Not Rec", "Rec"], yticklabels=["Expert Reject", "Expert Accept"])
        axes[1, 1].set_title("Decision Consistency")

        plt.tight_layout()
        st.pyplot(fig)

    with tab4:
        st.subheader("📝 文本化关键数据与统计")
        st.markdown("- **样本总量**:  共有患者 & 陪诊员 样本数，以及测试集基本统计。")
        st.write(f"患者数量: {len(patients)}，陪诊员数量: {len(escorts)}")

        st.write("---")
        st.write("**整体分数统计（测试集）**")
        stats = test_df["score"].agg(["count", "mean", "median", "std", "min", "max"]).to_frame().T
        st.table(stats)

        st.write("**分项指标描述**")
        comp_stats = test_df[["ability", "attitude", "similarity", "active", "score"]].describe()
        st.dataframe(comp_stats)

        st.write("---")
        st.write("**推荐与专家标注分布**")
        dist = pd.DataFrame({
            "预测": test_df["pred"].value_counts().sort_index(),
            "专家标注": test_df["truth"].value_counts().sort_index(),
        }).fillna(0).astype(int)
        st.table(dist)

        st.write("---")
        st.write("**Top 5 按平均分排序的陪诊员**")
        escort_scores = test_df.groupby("escort_id")["score"].mean().sort_values(ascending=False).head(5)
        st.table(escort_scores.rename("avg_score").to_frame())

        with st.expander("查看测试集示例行（前20行）", expanded=False):
            st.dataframe(test_df.head(20))

    with tab5:
        st.subheader("⚙️ 当前配置参数（JSON）")
        cfg = st.session_state.config
        try:
            cfg_str = json.dumps(cfg, ensure_ascii=False, indent=2)
        except Exception:
            cfg_str = str(cfg)

        st.code(cfg_str, language="json")
        st.download_button("下载配置 (JSON)", data=cfg_str, file_name="config.json", mime="application/json")


if __name__ == "__main__":
    main()
