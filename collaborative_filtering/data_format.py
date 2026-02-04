"""
用户协同过滤数据格式定义
用于处理用户行为数据和商品排序
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class UserBehavior:
    """用户行为数据"""
    user_id: str
    item_id: str
    behavior_type: str  # click, cart, view, order
    timestamp: datetime
    score: float  # 行为权重
    
    def __post_init__(self):
        # 行为权重映射
        behavior_weights = {
            'click': 1.0,
            'view': 0.5,
            'cart': 3.0,
            'order': 5.0
        }
        if self.score is None:
            self.score = behavior_weights.get(self.behavior_type, 1.0)

@dataclass
class CollaborativeProfile:
    """用户协同过滤画像"""
    user_id: str
    similar_users: List[str]  # 相似用户ID列表
    similarity_scores: List[float]  # 对应的相似度分数
    preferred_items: List[str]  # 偏好商品列表
    behavior_history: List[UserBehavior]  # 行为历史
    
@dataclass
class ItemRankingRequest:
    """商品排序请求"""
    target_user_id: str
    candidate_items: List[str]  # 待排序商品候选集
    context: Optional[Dict] = None  # 上下文信息（时间、场景等）

@dataclass
class RankingResult:
    """排序结果"""
    user_id: str
    ranked_items: List[str]  # 排序后的商品ID
    scores: List[float]  # 对应的预测分数
    explanation: Optional[str] = None  # 排序解释（可选）