"""
集成到AlignXplore+框架
将协同过滤功能集成到现有的评估和训练流程中
"""

import json
import os
from typing import List, Dict, Optional
from pathlib import Path

from data_format import UserBehavior, ItemRankingRequest, CollaborativeProfile
from user_similarity import UserSimilarityCalculator
from preference_generator import CollaborativePreferenceGenerator
from item_ranking import CollaborativeItemRanker


class CollaborativeAlignXploreIntegration:
    """协同过滤与AlignXplore+集成器"""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.similarity_calculator = None
        self.preference_generator = None
        self.item_ranker = None
        
        # 确保目录存在
        self.data_path.mkdir(exist_ok=True)
        (self.data_path / "collaborative").mkdir(exist_ok=True)
        (self.data_path / "preferences").mkdir(exist_ok=True)
        (self.data_path / "results").mkdir(exist_ok=True)
    
    def prepare_collaborative_data(self, behavior_data_path: str) -> bool:
        """准备协同过滤数据"""
        try:
            # 1. 加载用户行为数据
            behaviors = self._load_behavior_data(behavior_data_path)
            
            # 2. 构建相似度计算器
            self.similarity_calculator = UserSimilarityCalculator(behaviors)
            
            # 3. 构建偏好生成器
            self.preference_generator = CollaborativePreferenceGenerator(
                self.similarity_calculator
            )
            
            # 4. 构建商品排序器
            self.item_ranker = CollaborativeItemRanker(self.preference_generator)
            
            print(f"成功加载 {len(behaviors)} 条行为数据")
            return True
            
        except Exception as e:
            print(f"数据准备失败: {e}")
            return False
    
    def generate_collaborative_preferences(self, output_path: str = None) -> str:
        """生成协同过滤偏好数据"""
        if not self.similarity_calculator:
            raise ValueError("请先调用prepare_collaborative_data")
        
        if output_path is None:
            output_path = self.data_path / "collaborative" / "collaborative_preferences.json"
        
        # 获取所有用户
        all_users = set()
        for behavior in self.similarity_calculator.behavior_data:
            all_users.add(behavior.user_id)
        
        collaborative_data = []
        
        for user_id in all_users:
            # 构建用户画像
            user_profile = self.similarity_calculator.build_collaborative_profile(
                user_id, top_k=10
            )
            
            # 生成偏好描述
            preference_desc = self.preference_generator.generate_collaborative_preference(
                user_profile
            )
            
            # 构建AlignXplore+格式
            collaborative_item = {
                "user_id": user_id,
                "profile": preference_desc,
                "similar_users": user_profile.similar_users,
                "similarity_scores": user_profile.similarity_scores,
                "preferred_items": user_profile.preferred_items,
                "behavior_history": [
                    {
                        "item_id": b.item_id,
                        "behavior_type": b.behavior_type,
                        "timestamp": b.timestamp.isoformat(),
                        "score": b.score
                    }
                    for b in user_profile.behavior_history
                ]
            }
            
            collaborative_data.append(collaborative_item)
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(collaborative_data, f, ensure_ascii=False, indent=2)
        
        print(f"协同过滤偏好数据已保存到: {output_path}")
        return str(output_path)
    
    def convert_to_alignxplore_format(self, input_file: str, output_file: str,
                                    task_type: str = "recommendation") -> str:
        """转换为AlignXplore+评估格式"""
        
        # 加载协同过滤数据
        with open(input_file, 'r', encoding='utf-8') as f:
            collaborative_data = json.load(f)
        
        alignxplore_data = []
        
        for item in collaborative_data:
            user_id = item["user_id"]
            profile = item["profile"]
            
            if task_type == "recommendation":
                # 推荐任务格式
                alignxplore_item = {
                    "history": item["behavior_history"],
                    "profile": profile,
                    "target": {
                        "query": f"为用户{user_id}推荐商品",
                        "candidates": item["preferred_items"][:10]  # 候选商品
                    }
                }
            else:
                # 选择任务格式
                candidates = item["preferred_items"][:10]
                if len(candidates) >= 2:
                    alignxplore_item = {
                        "history": item["behavior_history"],
                        "profile": profile,
                        "target": {
                            "query": "选择更合适的商品",
                            "chosen": candidates[0],
                            "rejected": candidates[1] if len(candidates) > 1 else candidates[0]
                        }
                    }
                else:
                    continue
            
            alignxplore_data.append(alignxplore_item)
        
        # 保存转换后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(alignxplore_data, f, ensure_ascii=False, indent=2)
        
        print(f"AlignXplore+格式数据已保存到: {output_file}")
        return output_file
    
    def rank_items_for_user(self, user_id: str, candidate_items: List[str]) -> Dict:
        """为特定用户排序商品"""
        if not self.item_ranker:
            raise ValueError("请先调用prepare_collaborative_data")
        
        # 构建用户画像
        user_profile = self.similarity_calculator.build_collaborative_profile(user_id)
        
        # 构建排序请求
        request = ItemRankingRequest(
            target_user_id=user_id,
            candidate_items=candidate_items
        )
        
        # 执行排序
        ranking_result = self.item_ranker.rank_items(request, user_profile)
        
        # 转换为字典格式
        result = {
            "user_id": ranking_result.user_id,
            "ranked_items": ranking_result.ranked_items,
            "scores": ranking_result.scores,
            "explanation": ranking_result.explanation
        }
        
        return result
    
    def evaluate_collaborative_performance(self, test_data_path: str,
                                         output_dir: str = None) -> Dict:
        """评估协同过滤性能"""
        if output_dir is None:
            output_dir = self.data_path / "results" / "collaborative"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载测试数据
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        results = []
        metrics = {
            "total_users": 0,
            "successful_rankings": 0,
            "average_ranking_score": 0.0
        }
        
        for test_item in test_data:
            user_id = test_item.get("user_id")
            candidate_items = test_item.get("candidate_items", [])
            expected_items = test_item.get("expected_items", [])
            
            if not user_id or not candidate_items:
                continue
            
            metrics["total_users"] += 1
            
            try:
                # 执行排序
                ranking_result = self.rank_items_for_user(user_id, candidate_items)
                
                # 计算指标
                result_item = {
                    "user_id": user_id,
                    "ranked_items": ranking_result["ranked_items"],
                    "scores": ranking_result["scores"],
                    "expected_items": expected_items,
                    "hit_rate": self._calculate_hit_rate(
                        ranking_result["ranked_items"], expected_items
                    ),
                    "ndcg": self._calculate_ndcg(
                        ranking_result["ranked_items"], expected_items
                    )
                }
                
                results.append(result_item)
                metrics["successful_rankings"] += 1
                
            except Exception as e:
                print(f"用户 {user_id} 排序失败: {e}")
                continue
        
        # 计算总体指标
        if results:
            metrics["average_hit_rate"] = np.mean([r["hit_rate"] for r in results])
            metrics["average_ndcg"] = np.mean([r["ndcg"] for r in results])
            metrics["average_ranking_score"] = np.mean([
                np.mean(r["scores"][:10]) for r in results if r["scores"]
            ])
        
        # 保存详细结果
        results_file = output_dir / "collaborative_evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metrics": metrics,
                "detailed_results": results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"评估结果已保存到: {results_file}")
        return metrics
    
    def _load_behavior_data(self, data_path: str) -> List[UserBehavior]:
        """加载行为数据"""
        behaviors = []
        
        # 支持多种数据格式
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    behavior = UserBehavior(
                        user_id=item["user_id"],
                        item_id=item["item_id"],
                        behavior_type=item["behavior_type"],
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        score=item.get("score")
                    )
                    behaviors.append(behavior)
        
        elif data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    behavior = UserBehavior(
                        user_id=item["user_id"],
                        item_id=item["item_id"],
                        behavior_type=item["behavior_type"],
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        score=item.get("score")
                    )
                    behaviors.append(behavior)
        
        return behaviors
    
    def _calculate_hit_rate(self, ranked_items: List[str], expected_items: List[str],
                          k: int = 10) -> float:
        """计算Hit Rate@K"""
        if not expected_items:
            return 0.0
        
        top_k_items = set(ranked_items[:k])
        expected_set = set(expected_items)
        
        hits = len(top_k_items.intersection(expected_set))
        return hits / len(expected_items)
    
    def _calculate_ndcg(self, ranked_items: List[str], expected_items: List[str],
                       k: int = 10) -> float:
        """计算NDCG@K"""
        if not expected_items:
            return 0.0
        
        # 构建相关性评分（二值）
        relevance = []
        for item in ranked_items[:k]:
            relevance.append(1.0 if item in expected_items else 0.0)
        
        # 计算DCG
        dcg = 0.0
        for i, rel in enumerate(relevance):
            dcg += rel / np.log2(i + 2)  # i+2因为log2(1)=0
        
        # 计算IDCG（理想DCG）
        ideal_relevance = sorted([1.0] * len(expected_items) + [0.0] * (k - len(expected_items)),
                               reverse=True)
        
        idcg = 0.0
        for i, rel in enumerate(ideal_relevance):
            idcg += rel / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0


# 集成到现有评估脚本的适配器
class CollaborativeEvaluationAdapter:
    """协同过滤评估适配器"""
    
    def __init__(self, integration: CollaborativeAlignXploreIntegration):
        self.integration = integration
    
    def adapt_for_evaluate_rec_pair(self, collaborative_data_file: str,
                                  output_file: str) -> str:
        """适配到evaluate_rec_pair.py格式"""
        
        with open(collaborative_data_file, 'r', encoding='utf-8') as f:
            collaborative_data = json.load(f)
        
        eval_data = []
        
        for item in collaborative_data:
            # 构建成对比较数据
            preferred_items = item["preferred_items"][:10]
            
            for i in range(0, len(preferred_items) - 1, 2):
                if i + 1 < len(preferred_items):
                    eval_item = {
                        "history": item["behavior_history"][:5],  # 取最近5个行为
                        "profile": item["profile"],
                        "target": {
                            "query": f"为用户{item['user_id']}推荐商品",
                            "chosen": preferred_items[i],
                            "rejected": preferred_items[i + 1]
                        }
                    }
                    eval_data.append(eval_item)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)
        
        return output_file
    
    def adapt_for_evaluate_select(self, collaborative_data_file: str,
                                output_file: str) -> str:
        """适配到evaluate_select.py格式"""
        
        # 选择任务和推荐任务使用相同的格式
        return self.adapt_for_evaluate_rec_pair(collaborative_data_file, output_file)