import pandas as pd
import glob
import os
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score

def main():
    # ================= 1. 定义命令行参数 =================
    parser = argparse.ArgumentParser(description="计算跨生成器的平均 AUC 和 AP 指标")
    parser.add_argument(
        "--scores-dir", 
        type=str, 
        required=True, 
        help="存放 infer 命令输出的真假图片分数 CSV 的目录路径"
    )
    parser.add_argument(
        "--output-csv", 
        type=str, 
        required=True, 
        help="最终汇总结果表格的保存路径 (包含文件名，如 ./result.csv)"
    )
    parser.add_argument(
        "--tag-column", 
        type=str, 
        required=True, 
        help="最终表格中标识生成器得分的列名"
    )
    
    # 解析外部传入的参数
    args = parser.parse_args()
    SCORES_DIR = args.scores_dir
    OUTPUT_CSV = args.output_csv
    TAG_COLUMN = args.tag_column
    print(f"📂 数据读取目录: {SCORES_DIR}")
    print(f"💾 结果保存路径: {OUTPUT_CSV}")
    print(f"🏷️ 标签列名: {TAG_COLUMN}\n")
    # ====================================================

    # 2. 加载 5 个 real 子集的分数
    real_files = glob.glob(os.path.join(SCORES_DIR, "real_*.csv"))
    if not real_files:
        print(f"❌ 错误：在 {SCORES_DIR} 目录下没有找到任何 real_*.csv 文件！")
        return

    real_scores_dict = {}
    for real_file in real_files:
        dataset_name = os.path.basename(real_file).replace(".csv", "")
        df = pd.read_csv(real_file) 
        # ⚠️ 记得根据实际情况核对这里的 'score' 列名
        real_scores_dict[dataset_name] = df[TAG_COLUMN].tolist() 

    print(f"✅ 成功加载了 {len(real_scores_dict)} 个真实图像子集的分数。\n")

    # 打印终端表头
    print(f"{'生成器 (Dataset)':<20} | {'Avg AUC (%)':<12} | {'Avg AP (%)':<12}")
    print("-" * 52)

    final_results_list = []

    # 3. 遍历所有 fake 分数文件
    fake_files = glob.glob(os.path.join(SCORES_DIR, "fake_*.csv"))

    for fake_file in fake_files:
        fake_name = os.path.basename(fake_file).replace(".csv", "").replace("fake_", "")
        
        df = pd.read_csv(fake_file)
        fake_scores = df[TAG_COLUMN].tolist()
        y_true_fake = [1] * len(fake_scores)
        
        auc_list = []
        ap_list = []
        
        # 4. 让当前假图分别与 5 个真图子集计算指标
        for real_name, r_scores in real_scores_dict.items():
            y_true_real = [0] * len(r_scores)
            
            y_true_combined = y_true_real + y_true_fake
            y_pred_combined = r_scores + fake_scores
            
            auc = roc_auc_score(y_true_combined, y_pred_combined)
            ap = average_precision_score(y_true_combined, y_pred_combined)
            
            auc_list.append(auc)
            ap_list.append(ap)
            
        # 5. 求算术平均值并转为百分制
        avg_auc = np.mean(auc_list) * 100
        avg_ap = np.mean(ap_list) * 100
        
        # 终端打印结果
        print(f"{fake_name:<20} | {avg_auc:>8.2f}     | {avg_ap:>8.2f}")
        
        final_results_list.append({
            "Generator": fake_name,
            "Avg_AUC(%)": round(avg_auc, 2),
            "Avg_AP(%)": round(avg_ap, 2)
        })

    # 6. 写入 CSV 逻辑
    if final_results_list:
        results_df = pd.DataFrame(final_results_list)
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        results_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        print(f"\n🎉 评估全部完成！最终成绩单已保存至: {OUTPUT_CSV}")
    else:
        print(f"\n⚠️ 警告：没有找到任何 fake_*.csv 文件，无法生成评估结果。")

if __name__ == "__main__":
    main()