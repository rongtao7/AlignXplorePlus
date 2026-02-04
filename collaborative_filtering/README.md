# 协同过滤推荐模块

基于AlignXplore+框架的用户协同过滤实现，支持基于相似用户的点击/加购/浏览/下单行为进行商品排序推荐。

## 🎯 功能特性

- **多维度相似度计算**：支持Jaccard相似度、余弦相似度、行为模式相似度
- **智能偏好生成**：基于相似用户行为自动生成自然语言偏好描述
- **商品排序算法**：综合协同过滤分数和偏好匹配度进行排序
- **AlignXplore+集成**：无缝集成到现有评估和训练框架
- **流式处理支持**：支持大规模数据的实时处理
- **可解释推荐**：提供推荐结果的详细解释

## 📦 安装依赖

```bash
pip install numpy scikit-learn pandas
```

## 🚀 快速开始

### 1. 基础使用

```python
from collaborative_filtering import CollaborativeAlignXploreIntegration

# 初始化系统
cf_integration = CollaborativeAlignXploreIntegration(data_path="data")

# 准备数据（JSON格式）
cf_integration.prepare_collaborative_data("behavior_data.json")

# 为特定用户排序商品
result = cf_integration.rank_items_for_user(
    user_id="user_001",
    candidate_items=["item_101", "item_102", "item_103"]
)

print(f"推荐结果: {result['ranked_items']}")
print(f"推荐分数: {result['scores']}")
```

### 2. 数据格式

行为数据JSON格式：
```json
[
  {
    "user_id": "user_001",
    "item_id": "item_101",
    "behavior_type": "click",
    "timestamp": "2024-01-15T10:30:00",
    "score": 1.0
  }
]
```

支持的行为类型及权重：
- `click`: 1.0 （点击）
- `view`: 0.5 （浏览）
- `cart`: 3.0 （加购）
- `order`: 5.0 （下单）

### 3. 生成AlignXplore+格式数据

```python
# 生成协同过滤偏好
pref_file = cf_integration.generate_collaborative_preferences()

# 转换为评估格式
eval_file = cf_integration.convert_to_alignxplore_format(
    pref_file, 
    "eval_data.json",
    task_type="recommendation"  # 或 "selection"
)
```

## 📊 核心算法

### 用户相似度计算

综合多种相似度指标：
- **Jaccard相似度**：基于商品交互集合的相似性
- **余弦相似度**：基于评分向量的相似性  
- **行为模式相似度**：基于用户行为类型分布的相似性

```python
similarity = calculator.combined_similarity(
    user1, user2,
    weights={'jaccard': 0.3, 'cosine': 0.4, 'behavior': 0.3}
)
```

### 偏好描述生成

自动生成自然语言偏好描述：
```python
preference_desc = generator.generate_collaborative_preference(user_profile)
# 输出: "基于与用户user_001相似的3个用户的偏好分析：偏好商品类别：电子产品、服装..."
```

### 商品排序算法

综合评分机制：
- **协同过滤分数** (60%): 基于相似用户的交互数据
- **偏好匹配度** (40%): 基于商品特征与偏好描述的相关性

## 🔧 高级功能

### 批量处理

```python
# 批量排序请求
requests = [
    {"user_id": "user_001", "candidate_items": [...]},
    {"user_id": "user_002", "candidate_items": [...]},
]

results = ranker.batch_rank_items(requests, user_profiles)
```

### 性能评估

```python
# 评估协同过滤性能
metrics = cf_integration.evaluate_collaborative_performance(
    "test_data.json",
    output_dir="results"
)

print(f"平均Hit Rate: {metrics['average_hit_rate']}")
print(f"平均NDCG: {metrics['average_ndcg']}")
```

## 📁 文件结构

```
collaborative_filtering/
├── __init__.py                 # 模块初始化
├── data_format.py             # 数据格式定义
├── user_similarity.py         # 用户相似度计算
├── preference_generator.py    # 偏好描述生成
├── item_ranking.py           # 商品排序算法
├── integrate_alignxplore.py  # AlignXplore+集成
├── example_usage.py          # 使用示例
├── test_integration.py       # 测试用例
└── README.md                 # 本文档
```

## 🧪 测试和验证

运行测试用例：
```bash
python collaborative_filtering/test_integration.py
```

运行使用示例：
```bash
python collaborative_filtering/example_usage.py
```

## 📈 性能指标

支持的标准推荐系统评估指标：
- **Hit Rate@K**: K个推荐中的命中率
- **NDCG@K**: 归一化折损累积增益
- **平均排序分数**: 推荐商品的质量评分

## 🔗 与AlignXplore+集成

该模块完全兼容现有的AlignXplore+评估框架：

```bash
# 生成评估数据
python collaborative_filtering/example_usage.py

# 使用现有评估脚本
python eval/evaluate_rec_pair.py --input_file data/rec_eval_data.json --model_name Qwen/Qwen3-8B
python eval/evaluate_select.py --input_file data/select_eval_data.json --model_name Qwen/Qwen3-8B
```

## 💡 使用场景

- **电商推荐**: 基于用户购买、浏览、加购行为
- **内容推荐**: 基于用户点击、阅读、收藏行为  
- **社交网络**: 基于用户互动、关注行为
- **跨域迁移**: 支持不同领域间的偏好迁移

## ⚙️ 配置参数

主要可调参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| top_k | 10 | 相似用户数量 |
| min_common_items | 3 | 最小共同商品数 |
| collaborative_weight | 0.6 | 协同过滤分数权重 |
| preference_weight | 0.4 | 偏好匹配权重 |

## 📝 注意事项

1. **数据质量**: 确保行为数据的时间序列正确
2. **冷启动**: 新用户需要足够的行为数据才能获得准确推荐
3. **稀疏性**: 对于稀疏数据，建议调整`min_common_items`参数
4. **实时性**: 支持增量更新，可定期重新计算相似度

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个协同过滤模块！

## 📄 许可证

与AlignXplore+项目保持一致的开源许可证。