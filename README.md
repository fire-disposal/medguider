# 老年陪诊推荐系统 (pyV)

基于多维度画像建模的老年陪诊员推荐算法实验项目。

## 🚀 快速开始

本项目使用 [uv](https://github.com/astral-sh/uv) 管理依赖。

### 运行交互式看板 (推荐)
```bash
uv run streamlit run app.py
```

### 运行命令行模拟
```bash
uv run python main.py
```

## 📂 项目结构
- [src/pyv/](src/pyv/): 核心算法包 (Models, Engine, Config)
- [app.py](app.py): Streamlit 交互化前端应用
- [mainflow/](mainflow/): 历史脚本与模拟流程
- [.github/workflows/](.github/workflows/): CI 配置文件

## 🛠️ 核心功能
1. **多维度画像**: 集成患者自填量表、陪诊员证书、服务评分、语义匹配。
2. **工程化引擎**: 支持加权因子调整与专家标准对比评估。
3. **实时看板**: 提供 P-R 曲线、相关性热图及个案深度解析。
