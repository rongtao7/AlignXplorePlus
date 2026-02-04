"""
协同过滤推荐模块
基于AlignXplore+框架的用户协同过滤实现
"""

from .data_format import (
    UserBehavior,
    CollaborativeProfile,
    ItemRankingRequest,
    RankingResult
)

from .user_similarity import UserSimilarityCalculator

from .preference_generator import CollaborativePreferenceGenerator

from .item_ranking import CollaborativeItemRanker

from .integrate_alignxplore import (
    CollaborativeAlignXploreIntegration,
    CollaborativeEvaluationAdapter
)

__version__ = "1.0.0"
__author__ = "AlignXplore+ Team"

__all__ = [
    # 数据格式
    "UserBehavior",
    "CollaborativeProfile", 
    "ItemRankingRequest",
    "RankingResult",
    
    # 核心组件
    "UserSimilarityCalculator",
    "CollaborativePreferenceGenerator",
    "CollaborativeItemRanker",
    
    # 集成组件
    "CollaborativeAlignXploreIntegration",
    "CollaborativeEvaluationAdapter"
]