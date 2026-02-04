"""
协同过滤偏好生成器
基于相似用户生成目标用户的偏好描述
"""

import json
from typing import List, Dict, Optional
from collections import defaultdict
from datetime import datetime

from data_format import CollaborativeProfile, UserBehavior
from user_similarity import UserSimilarityCalculator


class CollaborativePreferenceGenerator:
    """协同过滤偏好生成器"""
    
    def __init__(self, similarity_calculator: UserSimilarityCalculator):
        self.similarity_calculator = similarity_calculator
    
    def generate_collaborative_preference(self, user_profile: CollaborativeProfile) -> str:
        """生成协同过滤偏好描述"""
        
        # 1. 分析相似用户的偏好模式
        similar_user_patterns = self._analyze_similar_user_patterns(user_profile)
        
        # 2. 生成偏好描述
        preference_description = self._build_preference_description(
            user_profile, similar_user_patterns
        )
        
        return preference_description
    
    def _analyze_similar_user_patterns(self, user_profile: CollaborativeProfile) -> Dict:
        """分析相似用户的偏好模式"""
        patterns = {
            'popular_categories': [],
            'preferred_brands': [],
            'price_preferences': [],
            'behavior_patterns': defaultdict(int),
            'item_features': defaultdict(list)
        }
        
        # 获取相似用户的行为数据
        similar_users_data = []
        for similar_user, similarity_score in zip(user_profile.similar_users, 
                                                 user_profile.similarity_scores):
            # 这里需要从原始数据中获取相似用户的行为
            # 假设我们可以通过某种方式获取
            user_behaviors = self._get_user_behaviors(similar_user)
            
            for behavior in user_behaviors:
                # 统计行为模式
                patterns['behavior_patterns'][behavior.behavior_type] += 1
                
                # 提取商品特征（需要根据实际商品数据）
                item_features = self._extract_item_features(behavior.item_id)
                for feature, value in item_features.items():
                    patterns['item_features'][feature].append(value)
        
        # 汇总统计
        patterns['popular_categories'] = self._get_most_frequent(
            patterns['item_features'].get('category', [])
        )
        patterns['preferred_brands'] = self._get_most_frequent(
            patterns['item_features'].get('brand', [])
        )
        patterns['price_preferences'] = self._analyze_price_preference(
            patterns['item_features'].get('price', [])
        )
        
        return patterns
    
    def _build_preference_description(self, user_profile: CollaborativeProfile,
                                    patterns: Dict) -> str:
        """构建偏好描述文本"""
        
        descriptions = []
        
        # 基础信息
        descriptions.append(f"基于与用户{user_profile.user_id}相似的{len(user_profile.similar_users)}个用户的偏好分析：")
        
        # 商品类别偏好
        if patterns['popular_categories']:
            top_categories = patterns['popular_categories'][:3]
            descriptions.append(f"偏好商品类别：{', '.join(top_categories)}")
        
        # 品牌偏好
        if patterns['preferred_brands']:
            top_brands = patterns['preferred_brands'][:3]
            descriptions.append(f"偏好品牌：{', '.join(top_brands)}")
        
        # 价格偏好
        if patterns['price_preferences']:
            price_desc = self._describe_price_preference(patterns['price_preferences'])
            descriptions.append(f"价格偏好：{price_desc}")
        
        # 行为模式
        if patterns['behavior_patterns']:
            behavior_desc = self._describe_behavior_pattern(patterns['behavior_patterns'])
            descriptions.append(f"行为特征：{behavior_desc}")
        
        # 协同过滤推荐商品
        if user_profile.preferred_items:
            top_items = user_profile.preferred_items[:5]
            descriptions.append(f"基于相似用户推荐：{', '.join(top_items)}")
        
        return "\n".join(descriptions)
    
    def generate_alignxplore_format(self, user_profile: CollaborativeProfile,
                                  candidate_items: List[str]) -> Dict:
        """生成AlignXplore+所需格式"""
        
        # 生成偏好描述
        preference_text = self.generate_collaborative_preference(user_profile)
        
        # 构建商品对比较任务
        item_pairs = []
        for i in range(len(candidate_items)):
            for j in range(i+1, len(candidate_items)):
                item_pairs.append({
                    'item_a': candidate_items[i],
                    'item_b': candidate_items[j],
                    'preference': preference_text
                })
        
        # 构建推荐任务
        recommendation_task = {
            'user_id': user_profile.user_id,
            'preference_summary': preference_text,
            'candidate_items': candidate_items,
            'item_pairs': item_pairs,
            'history': self._format_user_history(user_profile.behavior_history)
        }
        
        return recommendation_task
    
    def _format_user_history(self, behaviors: List[UserBehavior]) -> str:
        """格式化用户历史行为"""
        if not behaviors:
            return "暂无历史行为记录"
        
        # 按时间排序
        sorted_behaviors = sorted(behaviors, key=lambda x: x.timestamp, reverse=True)
        
        # 取最近的行为
        recent_behaviors = sorted_behaviors[:10]
        
        history_items = []
        for behavior in recent_behaviors:
            item_info = f"{behavior.item_id}({behavior.behavior_type})"
            history_items.append(item_info)
        
        return f"最近行为：{', '.join(history_items)}"
    
    def _get_user_behaviors(self, user_id: str) -> List[UserBehavior]:
        """获取用户行为数据（需要从原始数据源获取）"""
        # 这里应该连接到实际的数据源
        # 暂时返回空列表，需要在实际实现中补充
        return []
    
    def _extract_item_features(self, item_id: str) -> Dict:
        """提取商品特征"""
        # 这里应该连接到商品数据库
        # 返回商品的特征信息
        return {
            'category': 'unknown',
            'brand': 'unknown',
            'price': 0.0
        }
    
    def _get_most_frequent(self, items: List, top_n: int = 5) -> List:
        """获取最频繁的项目"""
        if not items:
            return []
        
        counter = Counter(items)
        return [item for item, _ in counter.most_common(top_n)]
    
    def _analyze_price_preference(self, prices: List[float]) -> List[str]:
        """分析价格偏好"""
        if not prices:
            return []
        
        prices = [p for p in prices if p > 0]
        if not prices:
            return []
        
        avg_price = np.mean(prices)
        price_ranges = []
        
        if avg_price < 50:
            price_ranges.append("低价商品")
        elif avg_price < 200:
            price_ranges.append("中价商品")
        else:
            price_ranges.append("高价商品")
        
        return price_ranges
    
    def _describe_price_preference(self, price_preferences: List[str]) -> str:
        """描述价格偏好"""
        if not price_preferences:
            return "暂无明确价格偏好"
        
        return f"偏好{'、'.join(set(price_preferences))}"
    
    def _describe_behavior_pattern(self, behavior_patterns: Dict[str, int]) -> str:
        """描述行为模式"""
        if not behavior_patterns:
            return "暂无明确行为模式"
        
        # 按行为频率排序
        sorted_behaviors = sorted(behavior_patterns.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        behavior_names = {
            'click': '点击',
            'view': '浏览',
            'cart': '加购',
            'order': '下单'
        }
        
        descriptions = []
        for behavior, count in sorted_behaviors[:3]:
            behavior_name = behavior_names.get(behavior, behavior)
            descriptions.append(f"{behavior_name}({count}次)")
        
        return "、".join(descriptions)