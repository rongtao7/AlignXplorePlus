"""
商品排序算法
基于协同过滤和偏好描述对商品进行排序
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from data_format import ItemRankingRequest, RankingResult, CollaborativeProfile
from preference_generator import CollaborativePreferenceGenerator


class CollaborativeItemRanker:
    """协同过滤商品排序器"""
    
    def __init__(self, preference_generator: CollaborativePreferenceGenerator):
        self.preference_generator = preference_generator
    
    def rank_items(self, request: ItemRankingRequest, 
                   user_profile: CollaborativeProfile) -> RankingResult:
        """对商品进行排序"""
        
        # 1. 获取商品特征
        item_features = self._get_item_features(request.candidate_items)
        
        # 2. 基于协同过滤计算基础分数
        collaborative_scores = self._calculate_collaborative_scores(
            user_profile, request.candidate_items
        )
        
        # 3. 基于偏好描述计算相关性分数
        preference_scores = self._calculate_preference_scores(
            user_profile, item_features
        )
        
        # 4. 综合排序分数
        final_scores = self._combine_scores(collaborative_scores, preference_scores)
        
        # 5. 排序并生成结果
        ranked_items, scores = self._sort_items(request.candidate_items, final_scores)
        
        # 6. 生成解释
        explanation = self._generate_explanation(user_profile, ranked_items[:3])
        
        return RankingResult(
            user_id=request.target_user_id,
            ranked_items=ranked_items,
            scores=scores,
            explanation=explanation
        )
    
    def _get_item_features(self, item_ids: List[str]) -> Dict[str, Dict]:
        """获取商品特征"""
        item_features = {}
        
        for item_id in item_ids:
            # 这里应该连接到商品数据库获取特征
            # 模拟商品特征
            features = self._extract_item_features(item_id)
            item_features[item_id] = features
        
        return item_features
    
    def _extract_item_features(self, item_id: str) -> Dict:
        """提取商品特征（模拟实现）"""
        # 在实际应用中，这里应该连接到商品数据库
        # 模拟一些商品特征
        import hashlib
        
        # 基于item_id生成一致的模拟特征
        item_hash = int(hashlib.md5(item_id.encode()).hexdigest(), 16)
        
        categories = ['电子产品', '服装', '食品', '图书', '家居']
        brands = ['品牌A', '品牌B', '品牌C', '品牌D']
        
        return {
            'category': categories[item_hash % len(categories)],
            'brand': brands[item_hash % len(brands)],
            'price': 50 + (item_hash % 500),  # 50-550元
            'popularity': item_hash % 100,  # 受欢迎程度
            'quality_score': 0.7 + (item_hash % 30) / 100,  # 0.7-1.0
            'discount_rate': 0.1 + (item_hash % 20) / 100  # 0.1-0.3
        }
    
    def _calculate_collaborative_scores(self, user_profile: CollaborativeProfile,
                                      candidate_items: List[str]) -> Dict[str, float]:
        """计算协同过滤分数"""
        scores = {}
        
        # 基于相似用户的交互数据
        for item_id in candidate_items:
            score = 0.0
            
            # 如果商品在相似用户的偏好列表中
            if item_id in user_profile.preferred_items:
                # 获取该商品在偏好列表中的排名
                rank = user_profile.preferred_items.index(item_id)
                # 排名越靠前，分数越高
                position_score = 1.0 / (rank + 1)
                
                # 考虑相似度权重
                if user_profile.similarity_scores:
                    avg_similarity = np.mean(user_profile.similarity_scores)
                    score = position_score * avg_similarity
            
            scores[item_id] = score
        
        # 归一化
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {item: score/max_score for item, score in scores.items()}
        
        return scores
    
    def _calculate_preference_scores(self, user_profile: CollaborativeProfile,
                                   item_features: Dict[str, Dict]) -> Dict[str, float]:
        """计算偏好相关性分数"""
        scores = {}
        
        # 生成偏好描述
        preference_desc = self.preference_generator.generate_collaborative_preference(
            user_profile
        )
        
        for item_id, features in item_features.items():
            score = 0.0
            
            # 基于商品特征计算相关性
            # 1. 类别匹配
            if '偏好商品类别' in preference_desc:
                category_match = self._check_category_match(preference_desc, features['category'])
                score += category_match * 0.3
            
            # 2. 品牌匹配
            if '偏好品牌' in preference_desc:
                brand_match = self._check_brand_match(preference_desc, features['brand'])
                score += brand_match * 0.2
            
            # 3. 价格匹配
            if '价格偏好' in preference_desc:
                price_match = self._check_price_match(preference_desc, features['price'])
                score += price_match * 0.2
            
            # 4. 商品质量
            score += features['quality_score'] * 0.2
            
            # 5. 受欢迎程度
            score += (features['popularity'] / 100) * 0.1
            
            scores[item_id] = score
        
        # 归一化
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {item: score/max_score for item, score in scores.items()}
        
        return scores
    
    def _check_category_match(self, preference_desc: str, category: str) -> float:
        """检查类别匹配"""
        if category in preference_desc:
            return 1.0
        return 0.0
    
    def _check_brand_match(self, preference_desc: str, brand: str) -> float:
        """检查品牌匹配"""
        if brand in preference_desc:
            return 1.0
        return 0.0
    
    def _check_price_match(self, preference_desc: str, price: float) -> float:
        """检查价格匹配"""
        if '低价' in preference_desc and price < 100:
            return 1.0
        elif '中价' in preference_desc and 100 <= price < 300:
            return 1.0
        elif '高价' in preference_desc and price >= 300:
            return 1.0
        return 0.5  # 默认中等匹配
    
    def _combine_scores(self, collaborative_scores: Dict[str, float],
                       preference_scores: Dict[str, float]) -> Dict[str, float]:
        """综合分数计算"""
        final_scores = {}
        
        # 权重组合
        collaborative_weight = 0.6  # 协同过滤权重
        preference_weight = 0.4     # 偏好匹配权重
        
        all_items = set(collaborative_scores.keys()).union(set(preference_scores.keys()))
        
        for item_id in all_items:
            collab_score = collaborative_scores.get(item_id, 0.0)
            pref_score = preference_scores.get(item_id, 0.0)
            
            # 加权组合
            final_score = (collaborative_weight * collab_score + 
                          preference_weight * pref_score)
            
            final_scores[item_id] = final_score
        
        return final_scores
    
    def _sort_items(self, item_ids: List[str], scores: Dict[str, float]) -> Tuple[List[str], List[float]]:
        """对商品进行排序"""
        # 按分数排序
        sorted_items = sorted(
            [(item_id, scores.get(item_id, 0.0)) for item_id in item_ids],
            key=lambda x: x[1],
            reverse=True
        )
        
        ranked_items = [item_id for item_id, _ in sorted_items]
        item_scores = [score for _, score in sorted_items]
        
        return ranked_items, item_scores
    
    def _generate_explanation(self, user_profile: CollaborativeProfile,
                            top_items: List[str]) -> str:
        """生成排序解释"""
        explanations = []
        
        explanations.append(f"基于{len(user_profile.similar_users)}个相似用户的偏好为您推荐：")
        
        # 解释前3个商品
        for i, item_id in enumerate(top_items[:3], 1):
            explanations.append(f"{i}. {item_id} - 基于相似用户的高评分推荐")
        
        return "\n".join(explanations)
    
    def batch_rank_items(self, requests: List[ItemRankingRequest],
                        user_profiles: Dict[str, CollaborativeProfile]) -> List[RankingResult]:
        """批量商品排序"""
        results = []
        
        for request in requests:
            user_profile = user_profiles.get(request.target_user_id)
            if user_profile:
                result = self.rank_items(request, user_profile)
                results.append(result)
        
        return results