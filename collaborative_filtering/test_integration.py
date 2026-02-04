"""
协同过滤模块测试和验证
测试各个组件的功能和集成效果
"""

import unittest
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# 导入协同过滤模块
from collaborative_filtering import (
    UserBehavior,
    CollaborativeProfile,
    ItemRankingRequest,
    RankingResult,
    UserSimilarityCalculator,
    CollaborativePreferenceGenerator,
    CollaborativeItemRanker,
    CollaborativeAlignXploreIntegration
)


class TestDataFormat(unittest.TestCase):
    """测试数据格式类"""
    
    def test_user_behavior_creation(self):
        """测试用户行为创建"""
        behavior = UserBehavior(
            user_id="user_001",
            item_id="item_101",
            behavior_type="click",
            timestamp=datetime.now(),
            score=None
        )
        
        self.assertEqual(behavior.user_id, "user_001")
        self.assertEqual(behavior.item_id, "item_101")
        self.assertEqual(behavior.behavior_type, "click")
        self.assertEqual(behavior.score, 1.0)  # 默认分数
        
    def test_behavior_weights(self):
        """测试行为权重"""
        behaviors = [
            UserBehavior("user_001", "item_101", "click", datetime.now(), None),
            UserBehavior("user_001", "item_102", "view", datetime.now(), None),
            UserBehavior("user_001", "item_103", "cart", datetime.now(), None),
            UserBehavior("user_001", "item_104", "order", datetime.now(), None),
        ]
        
        expected_scores = [1.0, 0.5, 3.0, 5.0]
        for behavior, expected in zip(behaviors, expected_scores):
            self.assertEqual(behavior.score, expected)


class TestUserSimilarity(unittest.TestCase):
    """测试用户相似度计算"""
    
    def setUp(self):
        """设置测试数据"""
        self.behaviors = [
            UserBehavior("user_001", "item_101", "click", datetime.now(), 1.0),
            UserBehavior("user_001", "item_102", "view", datetime.now(), 0.5),
            UserBehavior("user_001", "item_103", "cart", datetime.now(), 3.0),
            UserBehavior("user_002", "item_101", "click", datetime.now(), 1.0),
            UserBehavior("user_002", "item_102", "click", datetime.now(), 1.0),
            UserBehavior("user_002", "item_104", "order", datetime.now(), 5.0),
            UserBehavior("user_003", "item_105", "view", datetime.now(), 0.5),
            UserBehavior("user_003", "item_106", "click", datetime.now(), 1.0),
        ]
        
        self.calculator = UserSimilarityCalculator(self.behaviors)
    
    def test_user_profile_building(self):
        """测试用户画像构建"""
        self.assertIn("user_001", self.calculator.user_item_matrix)
        self.assertIn("user_002", self.calculator.user_item_matrix)
        self.assertIn("user_003", self.calculator.user_item_matrix)
        
        # 检查用户1的商品
        user1_items = self.calculator.user_item_matrix["user_001"]
        self.assertIn("item_101", user1_items)
        self.assertIn("item_102", user1_items)
        self.assertIn("item_103", user1_items)
    
    def test_jaccard_similarity(self):
        """测试Jaccard相似度"""
        # user_001和user_002有共同商品item_101
        similarity = self.calculator.jaccard_similarity("user_001", "user_002")
        self.assertGreater(similarity, 0)
        self.assertLessEqual(similarity, 1.0)
        
        # user_001和user_003没有共同商品
        similarity = self.calculator.jaccard_similarity("user_001", "user_003")
        self.assertEqual(similarity, 0.0)
    
    def test_cosine_similarity(self):
        """测试余弦相似度"""
        similarity = self.calculator.cosine_similarity_score("user_001", "user_002")
        self.assertGreaterEqual(similarity, 0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_find_similar_users(self):
        """测试寻找相似用户"""
        similar_users = self.calculator.find_similar_users("user_001", top_k=2)
        
        self.assertLessEqual(len(similar_users), 2)
        for user, score in similar_users:
            self.assertIn(user, ["user_002", "user_003"])
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1.0)
    
    def test_collaborative_profile(self):
        """测试协同过滤画像构建"""
        profile = self.calculator.build_collaborative_profile("user_001", top_k=2)
        
        self.assertEqual(profile.user_id, "user_001")
        self.assertLessEqual(len(profile.similar_users), 2)
        self.assertEqual(len(profile.similar_users), len(profile.similarity_scores))


class TestPreferenceGenerator(unittest.TestCase):
    """测试偏好生成器"""
    
    def setUp(self):
        """设置测试数据"""
        self.behaviors = [
            UserBehavior("user_001", "item_101", "click", datetime.now(), 1.0),
            UserBehavior("user_001", "item_102", "view", datetime.now(), 0.5),
            UserBehavior("user_002", "item_101", "click", datetime.now(), 1.0),
            UserBehavior("user_002", "item_103", "cart", datetime.now(), 3.0),
        ]
        
        self.calculator = UserSimilarityCalculator(self.behaviors)
        self.generator = CollaborativePreferenceGenerator(self.calculator)
    
    def test_preference_generation(self):
        """测试偏好描述生成"""
        profile = CollaborativeProfile(
            user_id="user_001",
            similar_users=["user_002"],
            similarity_scores=[0.8],
            preferred_items=["item_103"],
            behavior_history=self.behaviors[:2]
        )
        
        preference_desc = self.generator.generate_collaborative_preference(profile)
        
        self.assertIsInstance(preference_desc, str)
        self.assertGreater(len(preference_desc), 0)
        self.assertIn("user_001", preference_desc)
    
    def test_alignxplore_format(self):
        """测试AlignXplore+格式生成"""
        profile = CollaborativeProfile(
            user_id="user_001",
            similar_users=["user_002"],
            similarity_scores=[0.8],
            preferred_items=["item_103", "item_104"],
            behavior_history=self.behaviors[:2]
        )
        
        candidate_items = ["item_103", "item_104", "item_105"]
        
        alignxplore_data = self.generator.generate_alignxplore_format(profile, candidate_items)
        
        self.assertIn("user_id", alignxplore_data)
        self.assertIn("preference_summary", alignxplore_data)
        self.assertIn("candidate_items", alignxplore_data)
        self.assertIn("item_pairs", alignxplore_data)
        self.assertIn("history", alignxplore_data)


class TestItemRanking(unittest.TestCase):
    """测试商品排序"""
    
    def setUp(self):
        """设置测试数据"""
        self.behaviors = [
            UserBehavior("user_001", "item_101", "click", datetime.now(), 1.0),
            UserBehavior("user_001", "item_102", "view", datetime.now(), 0.5),
            UserBehavior("user_002", "item_101", "click", datetime.now(), 1.0),
            UserBehavior("user_002", "item_103", "cart", datetime.now(), 3.0),
            UserBehavior("user_002", "item_104", "order", datetime.now(), 5.0),
        ]
        
        self.calculator = UserSimilarityCalculator(self.behaviors)
        self.generator = CollaborativePreferenceGenerator(self.calculator)
        self.ranker = CollaborativeItemRanker(self.generator)
    
    def test_item_ranking(self):
        """测试商品排序"""
        profile = CollaborativeProfile(
            user_id="user_001",
            similar_users=["user_002"],
            similarity_scores=[0.8],
            preferred_items=["item_103", "item_104"],
            behavior_history=self.behaviors[:2]
        )
        
        request = ItemRankingRequest(
            target_user_id="user_001",
            candidate_items=["item_103", "item_104", "item_105"]
        )
        
        result = self.ranker.rank_items(request, profile)
        
        self.assertIsInstance(result, RankingResult)
        self.assertEqual(result.user_id, "user_001")
        self.assertEqual(len(result.ranked_items), 3)
        self.assertEqual(len(result.scores), 3)
        self.assertIsNotNone(result.explanation)
        
        # 检查排序结果（偏好商品应该排在前面）
        self.assertIn("item_103", result.ranked_items[:2])
        self.assertIn("item_104", result.ranked_items[:2])


class TestIntegration(unittest.TestCase):
    """测试集成功能"""
    
    def setUp(self):
        """设置测试环境"""
        self.test_data_path = "test_behavior_data.json"
        self.create_test_data()
        
        self.integration = CollaborativeAlignXploreIntegration(data_path="test_data")
    
    def create_test_data(self):
        """创建测试数据"""
        test_data = [
            {"user_id": "test_user_001", "item_id": "item_001", "behavior_type": "click", "timestamp": "2024-01-01T10:00:00"},
            {"user_id": "test_user_001", "item_id": "item_002", "behavior_type": "view", "timestamp": "2024-01-01T11:00:00"},
            {"user_id": "test_user_002", "item_id": "item_001", "behavior_type": "click", "timestamp": "2024-01-01T12:00:00"},
            {"user_id": "test_user_002", "item_id": "item_003", "behavior_type": "cart", "timestamp": "2024-01-01T13:00:00"},
            {"user_id": "test_user_003", "item_id": "item_004", "behavior_type": "order", "timestamp": "2024-01-01T14:00:00"},
        ]
        
        with open(self.test_data_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    def test_data_preparation(self):
        """测试数据准备"""
        success = self.integration.prepare_collaborative_data(self.test_data_path)
        self.assertTrue(success)
        
        self.assertIsNotNone(self.integration.similarity_calculator)
        self.assertIsNotNone(self.integration.preference_generator)
        self.assertIsNotNone(self.integration.item_ranker)
    
    def test_preference_generation(self):
        """测试偏好生成"""
        self.integration.prepare_collaborative_data(self.test_data_path)
        
        pref_file = self.integration.generate_collaborative_preferences()
        
        self.assertTrue(Path(pref_file).exists())
        
        # 检查生成的文件内容
        with open(pref_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.assertGreater(len(data), 0)
        for item in data:
            self.assertIn("user_id", item)
            self.assertIn("profile", item)
            self.assertIn("similar_users", item)
            self.assertIn("preferred_items", item)
    
    def test_item_ranking_integration(self):
        """测试商品排序集成"""
        self.integration.prepare_collaborative_data(self.test_data_path)
        
        user_id = "test_user_001"
        candidate_items = ["item_003", "item_004", "item_005"]
        
        result = self.integration.rank_items_for_user(user_id, candidate_items)
        
        self.assertIn("user_id", result)
        self.assertIn("ranked_items", result)
        self.assertIn("scores", result)
        self.assertIn("explanation", result)
        
        self.assertEqual(result["user_id"], user_id)
        self.assertEqual(len(result["ranked_items"]), len(candidate_items))
        self.assertEqual(len(result["scores"]), len(candidate_items))
    
    def test_format_conversion(self):
        """测试格式转换"""
        self.integration.prepare_collaborative_data(self.test_data_path)
        
        pref_file = self.integration.generate_collaborative_preferences()
        output_file = "test_alignxplore_format.json"
        
        result_file = self.integration.convert_to_alignxplore_format(
            pref_file, output_file
        )
        
        self.assertTrue(Path(result_file).exists())
        
        # 检查转换后的格式
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.assertGreater(len(data), 0)
        for item in data:
            self.assertIn("history", item)
            self.assertIn("profile", item)
            self.assertIn("target", item)
    
    def tearDown(self):
        """清理测试文件"""
        test_files = [
            self.test_data_path,
            "test_alignxplore_format.json",
            "test_data/test_behavior_data.json",
            "test_data/collaborative/collaborative_preferences.json"
        ]
        
        for file_path in test_files:
            if Path(file_path).exists():
                Path(file_path).unlink()


class TestPerformanceMetrics(unittest.TestCase):
    """测试性能指标计算"""
    
    def test_hit_rate_calculation(self):
        """测试Hit Rate计算"""
        integration = CollaborativeAlignXploreIntegration()
        
        # 测试用例1：部分匹配
        ranked_items = ["item_001", "item_002", "item_003", "item_004", "item_005"]
        expected_items = ["item_002", "item_004"]
        
        hit_rate = integration._calculate_hit_rate(ranked_items, expected_items, k=3)
        self.assertEqual(hit_rate, 0.5)  # 1个命中，总共2个期望项目
        
        # 测试用例2：完全匹配
        ranked_items = ["item_001", "item_002", "item_003"]
        expected_items = ["item_001", "item_002"]
        
        hit_rate = integration._calculate_hit_rate(ranked_items, expected_items, k=3)
        self.assertEqual(hit_rate, 1.0)  # 全部命中
        
        # 测试用例3：无匹配
        ranked_items = ["item_001", "item_002", "item_003"]
        expected_items = ["item_004", "item_005"]
        
        hit_rate = integration._calculate_hit_rate(ranked_items, expected_items, k=3)
        self.assertEqual(hit_rate, 0.0)  # 无命中
    
    def test_ndcg_calculation(self):
        """测试NDCG计算"""
        integration = CollaborativeAlignXploreIntegration()
        
        # 测试用例1：理想排序
        ranked_items = ["item_001", "item_002", "item_003"]
        expected_items = ["item_001", "item_002"]
        
        ndcg = integration._calculate_ndcg(ranked_items, expected_items, k=3)
        self.assertEqual(ndcg, 1.0)  # 理想排序
        
        # 测试用例2：部分相关
        ranked_items = ["item_001", "item_003", "item_002"]
        expected_items = ["item_002", "item_001"]
        
        ndcg = integration._calculate_ndcg(ranked_items, expected_items, k=3)
        self.assertLess(ndcg, 1.0)  # 非理想排序
        self.assertGreater(ndcg, 0.0)  # 但有部分相关


def run_comprehensive_test():
    """运行综合测试"""
    print("开始运行协同过滤模块综合测试...\n")
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加所有测试类
    test_classes = [
        TestDataFormat,
        TestUserSimilarity,
        TestPreferenceGenerator,
        TestItemRanking,
        TestIntegration,
        TestPerformanceMetrics
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果摘要
    print(f"\n=== 测试结果摘要 ===")
    print(f"运行测试数: {result.testsRun}")
    print(f"失败数: {len(result.failures)}")
    print(f"错误数: {len(result.errors)}")
    print(f"成功率: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)