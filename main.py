from pyv import CONFIG, EngineeringEngine, generate_engineering_data

def main():
    print("=== 老年陪诊推荐系统 v0.3 (CLI 运行版) ===")
    
    # 1. 生成数据
    patients, escorts = generate_engineering_data(600)
    engine = EngineeringEngine(CONFIG)
    
    # 2. 评估
    train_df, test_df, acc, prec = engine.evaluate_system(patients, escorts)
    
    print(f"模拟结果:")
    print(f"- 准确率: {acc:.4f}")
    print(f"- 精确率: {prec:.4f}")
    print("\n建议运行: streamlit run app.py 以查看完整交互看板。")

if __name__ == "__main__":
    main()
