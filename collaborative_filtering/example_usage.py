"""
协同过滤推荐系统使用示例
演示如何使用基于AlignXplore+的协同过滤功能
"""

import json
from datetime import datetime, timedelta
from collaborative_filtering import (
    UserBehavior,
    CollaborativeAlignXploreIntegration,
    CollaborativeEvaluationAdapter
)


def create_sample_behavior_data():
    """创建示例行为数据"""
    
    sample_data = []
    
    # 用户1的数据
    user1_behaviors = [
        {"user_id": "user_001", "item_id": "item_101", "behavior_type": "click", "timestamp": "2024-01-15T10:30:00"},
        {"user_id": "user_001", "item_id": "item_102", "behavior_type": "view", "timestamp": "2024-01-15T11:15:00"},
        {"user_id": "user_001", "item_id": "item_103", "behavior_type": "cart", "timestamp": "2024-01-15T14:20:00"},
        {"user_id": "user_001", "item_id": "item_104", "behavior_type": "order", "timestamp": "2024-01-15T16:45:00"},
        {"user_id": "user_001", "item_id": "item_105", "behavior_type": "click", "timestamp": "2024-01-16T09:30:00"},
    ]
    
    # 用户2的数据（与用户1有相似行为）
    user2_behaviors = [
        {"user_id": "user_002", "item_id": "item_101", "behavior_type": "click", "timestamp": "2024-01-15T11:00:00"},
        {"user_id": "user_002", "item_id": "item_102", "behavior_type": "click", "timestamp": "2024-01-15T12:30:00"},
        {"user_id": "user_002", "item_id": "item_103", "behavior_type": "cart", "timestamp": "2024-01-15T15:00:00"},
        {"user_id": "user_002", "item_id": "item_106", "behavior_type": "view", "timestamp": "2024-01-16T10:15:00"},
        {"user_id": "user_002", "item_id": "item_107", "behavior_type": "order", "timestamp": "2024-01-16T14:30:00"},
    ]
    
    # 用户3的数据
    user3_behaviors = [
        {"user_id": "user_003", "item_id": "item_108", "behavior_type": "view", "timestamp": "2024-01-15T13:20:00"},
        {"user_id": "user_003", "item_id": "item_109", "behavior_type": "click", "timestamp": "2024-01-15T16:00:00"},
        {"user_id": "user_003", "item_id": "item_110", "behavior_type": "cart", "timestamp": "2024-01-16T11:45:00"},
        {"user_id": "user_003", "item_id": "item_111", "behavior_type": "order", "timestamp": "2024-01-16T15:20:00"},
    ]
    
    # 用户4的数据（与用户1有相似行为）
    user4_behaviors = [
        {"user_id": "user_004", "item_id": "item_101", "behavior_type": "view", "timestamp": "2024-01-15T09:45:00"},
        {"user_id": "user_004", "item_id": "item_104", "behavior_type": "click", "timestamp": "2024-01-15T13:30:00"},
        {"user_id": "user_004", "item_id": "item_105", "behavior_type": "cart", "timestamp": "2024-01-16T08:20:00"},
        {"user_id": "user_004", "item_id": "item_112", "behavior_type": "order", "timestamp": "2024-01-16T16:10:00"},
    ]
    
    # 合并所有数据
    sample_data.extend(user1_behaviors)
    sample_data.extend(user2_behaviors)
    sample_data.extend(user3_behaviors)
    sample_data.extend(user4_behaviors)
    
    return sample_data


def demo_basic_usage():
    """基础使用演示"""
    print("=== 协同过滤推荐系统演示 ===\n")
    
    # 1. 创建示例数据
    print("1. 创建示例行为数据...")
    sample_data = create_sample_behavior_data()
    
    # 保存示例数据
    with open("sample_behavior_data.json", 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    # 2. 初始化协同过滤系统
    print("2. 初始化协同过滤系统...")
    cf_integration = CollaborativeAlignXploreIntegration(data_path="demo_data")
    
    # 3. 准备数据
    print("3. 加载和准备数据...")
    success = cf_integration.prepare_collaborative_data("sample_behavior_data.json")
    
    if not success:
        print("数据准备失败！")
        return
    
    # 4. 生成协同过滤偏好
    print("4. 生成协同过滤偏好...")
    pref_file = cf_integration.generate_collaborative_preferences()
    
    # 5. 为特定用户排序商品
    print("\n5. 为用户推荐商品...")
    target_user = "user_001"
    candidate_items = ["item_106", "item_107", "item_108", "item_109", "item_110", "item_111", "item_112"]
    
    ranking_result = cf_integration.rank_items_for_user(target_user, candidate_items)
    
    print(f"\n用户 {target_user} 的商品排序结果：")
    for i, (item, score) in enumerate(zip(ranking_result["ranked_items"], ranking_result["scores"]), 1):
        print(f"  {i}. {item} (分数: {score:.4f})")
    
    print(f"\n推荐解释：{ranking_result['explanation']}")
    
    # 6. 转换为AlignXplore+格式
    print("\n6. 转换为AlignXplore+评估格式...")
    alignxplore_file = cf_integration.convert_to_alignxplore_format(
        pref_file, "demo_data/alignxplore_format.json"
    )
    
    print(f"转换完成，文件保存至: {alignxplore_file}")


def demo_evaluation():
    """评估演示"""
    print("\n=== 协同过滤性能评估演示 ===\n")
    
    # 创建测试数据
    test_data = [
        {
            "user_id": "user_001",
            "candidate_items": ["item_106", "item_107", "item_108", "item_109", "item_110"],
            "expected_items": ["item_106", "item_107"]  # 期望推荐的商品
        },
        {
            "user_id": "user_002",
            "candidate_items": ["item_108", "item_109", "item_110", "item_111", "item_112"],
            "expected_items": ["item_108", "item_112"]
        }
    ]
    
    # 保存测试数据
    with open("test_data.json", 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # 初始化系统（复用之前的）
    cf_integration = CollaborativeAlignXploreIntegration(data_path="demo_data")
    cf_integration.prepare_collaborative_data("sample_behavior_data.json")
    
    # 执行评估
    print("执行协同过滤性能评估...")
    metrics = cf_integration.evaluate_collaborative_performance("test_data.json")
    
    print(f"\n评估结果：")
    print(f"  总用户数: {metrics['total_users']}")
    print(f"  成功排序数: {metrics['successful_rankings']}")
    print(f"  平均Hit Rate: {metrics.get('average_hit_rate', 0):.4f}")
    print(f"  平均NDCG: {metrics.get('average_ndcg', 0):.4f}")
    print(f"  平均排序分数: {metrics.get('average_ranking_score', 0):.4f}")


def demo_batch_processing():
    """批量处理演示"""
    print("\n=== 批量商品排序演示 ===\n")
    
    # 创建批量请求
    batch_requests = [
        {
            "user_id": "user_001",
            "candidate_items": ["item_106", "item_107", "item_108", "item_109", "item_110"]
        },
        {
            "user_id": "user_002",
            "candidate_items": ["item_108", "item_109", "item_110", "item_111", "item_112"]
        },
        {
            "user_id": "user_003",
            "candidate_items": ["item_101", "item_102", "item_103", "item_104", "item_105"]
        }
    ]
    
    # 初始化系统
    cf_integration = CollaborativeAlignXploreIntegration(data_path="demo_data")
    cf_integration.prepare_collaborative_data("sample_behavior_data.json")
    
    # 批量处理
    print("执行批量商品排序...")
    
    for request in batch_requests:
        user_id = request["user_id"]
        candidate_items = request["candidate_items"]
        
        result = cf_integration.rank_items_for_user(user_id, candidate_items)
        
        print(f"\n用户 {user_id} 的Top-3推荐：")
        for i, (item, score) in enumerate(zip(result["ranked_items"][:3], result["scores"][:3]), 1):
            print(f"  {i}. {item} (分数: {score:.4f})")


def demo_integration_with_alignxplore():
    """与AlignXplore+集成演示"""
    print("\n=== 与AlignXplore+框架集成演示 ===\n")
    
    # 初始化系统
    cf_integration = CollaborativeAlignXploreIntegration(data_path="demo_data")
    cf_integration.prepare_collaborative_data("sample_behavior_data.json")
    
    # 生成协同过滤偏好
    pref_file = cf_integration.generate_collaborative_preferences()
    
    # 创建评估适配器
    eval_adapter = CollaborativeEvaluationAdapter(cf_integration)
    
    # 适配到推荐评估格式
    rec_eval_file = eval_adapter.adapt_for_evaluate_rec_pair(
        pref_file, "demo_data/rec_eval_data.json"
    )
    
    # 适配到选择评估格式
    select_eval_file = eval_adapter.adapt_for_evaluate_select(
        pref_file, "demo_data/select_eval_data.json"
    )
    
    print(f"推荐评估数据: {rec_eval_file}")
    print(f"选择评估数据: {select_eval_file}")
    
    print("\n现在可以使用现有的evaluate_rec_pair.py和evaluate_select.py来评估协同过滤性能！")
    print("例如：")
    print(f"python eval/evaluate_rec_pair.py --input_file {rec_eval_file} --model_name Qwen/Qwen3-8B")


if __name__ == "__main__":
    print("开始协同过滤推荐系统演示...\n")
    
    # 1. 基础使用演示
    demo_basic_usage()
    
    # 2. 评估演示
    demo_evaluation()
    
    # 3. 批量处理演示
    demo_batch_processing()
    
    # 4. 与AlignXplore+集成演示
    demo_integration_with_alignxplore()
    
    print("\n=== 演示完成 ===")
    print("所有演示文件已保存到当前目录和demo_data/目录下")