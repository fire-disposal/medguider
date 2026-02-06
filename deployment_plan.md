# 老年陪诊推荐系统前端交互化部署计划 (Streamlit + FastAPI)

## 0. 项目背景
为了将 `simv0.1.py` 中的算法逻辑转化为可供非技术人员使用的实验性工具，本项目将采用 **Streamlit** 作为前端交互框架，并可选配合 **FastAPI** 作为后端服务引擎。

---

## 1. 技术栈选择
*   **前端交互**: Streamlit (轻量级、数据驱动、无需 HTML/JS 即可完成 UI 搭建)
*   **业务逻辑/API**: FastAPI (高性能、类型检查、OpenAPI 文档自动生成)
*   **核心算法**: 基于原有 `ResearchEngine` (Apriori 挖掘 + 多因子契合评分)
*   **可视化**: Matplotlib / Seaborn (集成到 Streamlit 看板中)

---

## 2. 预想项目结构
```text
pyV/
├── app_streamlit.py       # Streamlit 应用主程序 (入口)
├── backend_api.py         # (可选) FastAPI 后端接口
├── mainflow/
│   ├── engine.py          # 核心算法类 (从 simv0.1.py 抽离)
│   └── simv0.1.py         # 原始模拟实验脚本 (保留参考)
└── requirements.txt       # 新增依赖: streamlit, fastapi, uvicorn
```

---

## 3. 实施路线图

### 第一阶段：算法解耦 (Decoupling)
1.  从 `mainflow/simv0.1.py` 中提取 `ResearchEngine` 类及 `get_simulated_survey_data` 函数。
2.  修复可视化代码，使其能够以 `matplotlib.figure.Figure` 对象返回，而非直接调用 `plt.show()`，以便嵌入 Streamlit。

### 第二阶段：Streamlit 基础框架搭建
1.  **侧边栏**: 患者特征输入 (Selectbox/Multiselect)、算法参数调节 (min_support 滑块)。
2.  **主面板**: 
    *   统计摘要 (Metric 卡片)
    *   匹配排名 (Dataframe/Table)
    *   可视化看板 (Pyplot 集成)

### 第三阶段：功能实现
1.  **数据上传**: 支持上传自定义的问卷 Excel/CSV 文件。
2.  **实时计算**: 点击“运行分析”后，触发 `ResearchEngine` 的挖掘与匹配逻辑。
3.  **结果导出**: 支持将推荐列表下载为 Excel。

---

## 4. 关键交互界面逻辑 (Streamlit伪代码)
```python
import streamlit as st
from mainflow.engine import ResearchEngine

st.title("👵 老年陪诊智能匹配平台")

# 1. 参数设置
min_support = st.sidebar.slider("最小支持度", 0.05, 0.3, 0.15)

# 2. 患者特征选择
patient_tags = st.multiselect("请选择患者特征:", ["Age:80+", "ADL:严重受损", ...])

if st.button("开始匹配"):
    # 调用引擎逻辑
    engine = ResearchEngine(data)
    results = engine.recommend_best_escorts(patient_tags, pool)
    
    # 展示结果
    st.write(results)
    st.pyplot(engine.generate_dashboard_fig())
```

---

## 5. 依赖准备
运行前请确保安装以下包：
```bash
pip install streamlit fastapi uvicorn
```

## 6. 后续扩展方向
*   **数据库接入**: 将陪诊员信息存入 SQLite/PostgreSQL。
*   **多用户支持**: 增加登录认证功能。
*   **容器化**: 使用 Docker 封装 Streamlit + FastAPI 服务，一键部署到云服务器。
