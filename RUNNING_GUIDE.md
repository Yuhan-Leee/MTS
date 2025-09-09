# MTS 运行指南

## ✅ 系统状态
MTS系统已成功设置并测试通过！

## 🚀 运行选项

### 选项1：快速测试（推荐首次使用）
```bash
cd /data/home/zhanght/MTS
python scripts/quick_test.py
```
**时间**: ~1分钟
**功能**: 测试数据处理功能，处理3个学科的小样本

### 选项2：完整流水线（生产运行）
```bash
cd /data/home/zhanght/MTS
# 全局模型训练
python scripts/quick_start.py

# 分学科对比实验
python scripts/quick_start_per_subject.py
```
**时间**: ~数小时（取决于数据量和硬件）
**功能**: 完整的数据预处理、特征提取和模型训练

### 选项3：分步执行（推荐用于调试）
```bash
# 步骤1：数据预处理（~5分钟）
python scripts/preprocess_data.py \
  --input_dir /mnt/sharedata/ssd_large/common/datasets/cais___mmlu \
  --output_dir data/processed

# 步骤2：特征提取（~数小时）
python scripts/extract_features.py \
  --model_path /mnt/sharedata/ssd_large/common/LLMs/Llama-2-7b-chat-hf \
  --data_dir data/processed \
  --output_dir data/features \
  --batch_size 1

# 步骤3a：全局模型训练（~30分钟）
python scripts/train_models.py \
  --features_dir data/features \
  --output_dir outputs/results \
  --num_epochs 20

# 步骤3b：分学科模型训练与对比（~数小时）
python scripts/train_per_subject.py \
  --features_dir data/features \
  --output_dir outputs/per_subject \
  --num_epochs 20
```

### 选项4：分学科对比实验（新功能）
```bash
# 运行全局 vs 分学科模型对比
python scripts/train_per_subject.py \
  --features_dir data/features \
  --output_dir outputs/per_subject_comparison \
  --num_epochs 20
```

**功能对比**：
- **全局模型**：一个模型处理所有学科，参数少，训练快
- **分学科模型**：每个学科单独训练模型，参数多，但可能更精准

## 📊 预期结果

### 成功运行后，你将看到：

1. **数据预处理**：
   ```
   找到 57 个学科
   预处理完成!
   训练: 14042, 验证: 1395, 测试: 1533
   ```

2. **特征提取**：
   ```
   模型加载完成: /path/to/model
   特征提取完成: torch.Size([14042, 4096])
   ```

3. **模型训练**：
   ```
   Vanilla Accuracy: 0.4523, ECE: 0.1521
   Global Thermometer Accuracy: 0.4531, ECE: 0.0754
   ECE Improvement: 0.0767
   ```

4. **分学科对比实验**：
   ```
   📊 BASELINES:
   Method          Accuracy   ECE
   -----------------------------------
   Vanilla         0.4523     0.1521
   Oracle          0.4523     0.0623

   🎯 THERMOMETER MODELS:
   Method                  Accuracy   ECE        Params
   ----------------------------------------------------
   Global                  0.4531     0.0754     262,658
   Subject-wise            0.4542     0.0689     14,946,102

   📈 IMPROVEMENT ANALYSIS:
   Global vs Vanilla ECE improvement: 0.0767
   Subject-wise vs Vanilla ECE improvement: 0.0832
   Subject-wise vs Global ECE improvement: 0.0065
   Parameter increase ratio: 56.89x
   ```

## 📁 输出文件结构

```
MTS/
├── data/
│   ├── processed/           # 预处理后的数据
│   │   ├── train.jsonl      # 训练数据
│   │   ├── val.jsonl        # 验证数据
│   │   ├── test.jsonl       # 测试数据
│   │   └── dev_by_subject.json
│   └── features/            # 提取的特征
│       ├── train_features.pkl
│       ├── val_features.pkl
│       └── test_features.pkl
└── outputs/
    ├── results/
    │   ├── results.json      # 数值结果
    │   └── evaluation_report.json  # 详细报告
    └── per_subject_comparison/
        ├── comparison_results.json  # 对比结果
        ├── global/
        │   └── best_model.pth  # 全局模型
        └── subject_wise/
            ├── subject_wise_results.json  # 分学科结果
            └── subject_0/
                └── best_model.pth  # 学科0的模型
```

## 🔧 性能优化建议

### 如果运行太慢：

1. **减少批大小**：
   ```bash
   python scripts/extract_features.py --batch_size 1
   ```

2. **使用更小的模型**：
   ```bash
   # 修改 config/default.yaml 中的模型路径
   model:
     path: "/path/to/smaller/model"
   ```

3. **减少训练轮数**：
   ```bash
   python scripts/train_models.py --num_epochs 10
   ```

4. **使用数据子集**：
   ```bash
   # 只处理前10个学科进行测试
   # 修改预处理脚本中的 subject_dirs[:3] 为 subject_dirs[:10]
   ```

## 🎯 成功标准

系统成功运行的标准：

1. ✅ **导入测试通过**：所有模块导入成功
2. ✅ **数据预处理完成**：生成JSONL文件
3. ✅ **特征提取完成**：生成PKL文件
4. ✅ **全局模型训练完成**：生成结果JSON文件
5. ✅ **分学科模型训练完成**：生成对比结果JSON文件
6. ✅ **ECE改进**：Thermometer的ECE比Vanilla低0.05+
7. ✅ **对比分析完成**：能够比较全局vs分学科的效果

## 🚨 常见问题

### 问题1：内存不足
```bash
# 解决方案：减少批大小
python scripts/extract_features.py --batch_size 1
```

### 问题2：模型路径错误
```bash
# 检查模型路径是否存在
ls /mnt/sharedata/ssd_large/common/LLMs/Llama-2-7b-chat-hf
```

### 问题3：数据路径错误
```bash
# 检查数据路径是否存在
ls /mnt/sharedata/ssd_large/common/datasets/cais___mmlu
```

### 问题4：CUDA内存不足
```bash
# 解决方案：使用CPU
# 修改 config/default.yaml
model:
  device: "cpu"
```

## 📈 下一步

1. **立即运行**：`python scripts/quick_test.py`
2. **完整运行**：`python scripts/quick_start.py`
3. **查看结果**：检查 `outputs/results/` 目录
4. **分析结果**：查看 `results.json` 中的ECE改进

---

## 🎉 总结

MTS系统已经完全设置完成并可以运行！建议：

1. **新手**：先运行 `python scripts/quick_test.py` 验证系统
2. **常规使用**：运行 `python scripts/quick_start.py` 完整流水线
3. **调试**：使用分步执行来诊断问题

系统已经准备好进行5-shot温度校准实验！