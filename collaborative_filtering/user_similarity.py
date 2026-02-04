"""
用户相似度计算模块
基于用户行为数据计算用户间的相似度
"""

import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from data_format import UserBehavior, CollaborativeProfile


class UserSimilarityCalculator:
    """用户相似度计算器"""
    
    def __init__(self, behavior_data: List[UserBehavior]):
        self.behavior_data = behavior_data
        self.user_item_matrix = None
        self.user_behavior_stats = None
        self._build_user_profiles()
    
    def _build_user_profiles(self):
        """构建用户画像"""
        # 用户-商品交互矩阵
        user_items = defaultdict(lambda: defaultdict(float))
        # 用户行为统计
        user_stats = defaultdict(lambda: defaultdict(int))
        
        for behavior in self.behavior_data:
            user_items[behavior.user_id][behavior.item_id] += behavior.score
            user_stats[behavior.user_id][behavior.behavior_type] += 1
        
        self.user_item_matrix = dict(user_items)
        self.user_behavior_stats = dict(user_stats)
    
    def jaccard_similarity(self, user1: str, user2: str) -> float:
        """基于商品交互的Jaccard相似度"""
        items1 = set(self.user_item_matrix.get(user1, {}).keys())
        items2 = set(self.user_item_matrix.get(user2, {}).keys())
        
        if not items1 or not items2:
            return 0.0
        
        intersection = len(items1.intersection(items2))
        union = len(items1.union(items2))
        
        return intersection / union if union > 0 else 0.0
    
    def cosine_similarity_score(self, user1: str, user2: str) -> float:
        """基于评分向量的余弦相似度"""
        items1 = self.user_item_matrix.get(user1, {})
        items2 = self.user_item_matrix.get(user2, {})
        
        # 获取共同商品
        common_items = set(items1.keys()).intersection(set(items2.keys()))
        
        if not common_items:
            return 0.0
        
        # 构建评分向量
        scores1 = [items1[item] for item in common_items]
        scores2 = [items2[item] for item in common_items]
        
        # 计算余弦相似度
        vec1 = np.array(scores1).reshape(1, -1)
        vec2 = np.array(scores2).reshape(1, -1)
        
        similarity = cosine_similarity(vec1, vec2)[0][0]
        return similarity
    
    def behavior_pattern_similarity(self, user1: str, user2: str) -> float:
        """基于行为模式的相似度"""
        stats1 = self.user_behavior_stats.get(user1, {})
        stats2 = self.user_behavior_stats.get(user2, {})
        
        # 行为类型权重
        behavior_types = ['click', 'view', 'cart', 'order']
        
        # 构建行为向量
        vec1 = [stats1.get(bt, 0) for bt in behavior_types]
        vec2 = [stats2.get(bt, 0) for bt in behavior_types]
        
        # 标准化
        if sum(vec1) == 0 or sum(vec2) == 0:
            return 0.0
        
        vec1 = np.array(vec1) / sum(vec1)
        vec2 = np.array(vec2) / sum(vec2)
        
        # 计算余弦相似度
        similarity = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
        return similarity
    
    def combined_similarity(self, user1: str, user2: str, 
                           weights: Dict[str, float] = None) -> float:
        """综合相似度计算"""
        if weights is None:
            weights = {
                'jaccard': 0.3,
                'cosine': 0.4,
                'behavior': 0.3
            }
        
        jaccard_sim = self.jaccard_similarity(user1, user2)
        cosine_sim = self.cosine_similarity_score(user1, user2)
        behavior_sim = self.behavior_pattern_similarity(user1, user2)
        
        combined_score = (
            weights['jaccard'] * jaccard_sim +
            weights['cosine'] * cosine_sim +
            weights['behavior'] * behavior_sim
        )
        
        return combined_score
    
    def find_similar_users(self, target_user: str, top_k: int = 10,
                          min_common_items: int = 3) -> List[Tuple[str, float]]:
        """找到最相似的用户"""
        similarities = []
        
        # 获取所有用户
        all_users = set(self.user_item_matrix.keys())
        
        for user in all_users:
            if user == target_user:
                continue
            
            # 检查共同商品数量
            target_items = set(self.user_item_matrix.get(target_user, {}).keys())
            user_items = set(self.user_item_matrix.get(user, {}).keys())
            common_count = len(target_items.intersection(user_items))
            
            if common_count >= min_common_items:
                similarity = self.combined_similarity(target_user, user)
                similarities.append((user, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def build_collaborative_profile(self, user_id: str, top_k: int = 10) -> CollaborativeProfile:
        """构建用户协同过滤画像"""
        similar_users = self.find_similar_users(user_id, top_k)
        
        # 获取相似用户的偏好商品
        preferred_items = self._get_preferred_items_from_similar_users(
            similar_users, user_id
        )
        
        # 获取用户行为历史
        user_behaviors = [
            behavior for behavior in self.behavior_data 
            if behavior.user_id == user_id
        ]
        
        return CollaborativeProfile(
            user_id=user_id,
            similar_users=[user for user, _ in similar_users],
            similarity_scores=[score for _, score in similar_users],
            preferred_items=preferred_items,
            behavior_history=user_behaviors
        )
    
    def _get_preferred_items_from_similar_users(self, similar_users: List[Tuple[str, float]],
                                              target_user: str) -> List[str]:
        """从相似用户获取偏好商品"""
        target_items = set(self.user_item_matrix.get(target_user, {}).keys())
        
        item_scores = defaultdict(float)
        
        for similar_user, similarity_score in similar_users:
            user_items = self.user_item_matrix.get(similar_user, {})
            
            for item_id, rating in user_items.items():
                # 只推荐目标用户未交互过的商品
                if item_id not in target_items:
                    item_scores[item_id] += rating * similarity_score
        
        # 按分数排序
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [item_id for item_id, _ in sorted_items]