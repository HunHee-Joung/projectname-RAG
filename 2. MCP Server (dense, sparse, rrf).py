# retrieve.py
import os
import json
import logging
import time
from typing import List, Dict, Optional, Any
from datetime import datetime
from collections import OrderedDict, defaultdict

import numpy as np
from sklearn.preprocessing import minmax_scale
from qdrant_client import QdrantClient
from qdrant_client.models import *
from FlagEmbedding import BGEM3FlagModel
from fastmcp import FastMCP

# --- Qdrant 및 모델 공통 설정 ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "192.168.0.249")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docling_search")
MODEL_NAME = "BAAI/bge-m3"

# --- retrieve.py 전용 설정 ---
RETRIEVE_CONFIG = {
    # 검색 파라미터
    "k_multiplier": 3,
    "diversity_limit": 2,
    "search_ef": 32,
    
    # Context-Aware 경로는 RRF 점수만 100% 활용
    "context_path_weights": {"rrf": 1.0},
    # Simple 경로는 Dense와 Sparse 점수를 6:4로 조합
    "simple_path_weights": {"dense": 0.6, "sparse": 0.4},
    
    # 부모 문맥 재정렬 가중치(alpha) 동적 설정
    "parent_rerank_config": {
        "thresholds": [(0.7, 0.6), (0.3, 0.7)], 
        "default": 0.85
    },
}

# --- 로깅 설정 ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 타이머 헬퍼 클래스 추가 ---
class SimpleTimer:
    """'with' 구문으로 코드 블록의 실행 시간을 측정하고 로깅하는 컨텍스트 매니저"""
    def __init__(self, name: str, timings: Dict, log_on_exit: bool = False):
        self.name = name
        self.timings = timings
        self.log_on_exit = log_on_exit
    
    def __enter__(self):
        self.start = time.monotonic()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.monotonic() - self.start) * 1000
        self.timings[self.name] = duration_ms
        if self.log_on_exit:
            logger.info(f"    ⏱️  [{self.name}] 소요 시간: {duration_ms:.2f} ms")

# --- LRU 캐시 클래스 ---
class LRUCache(OrderedDict):
    def __init__(self, capacity: int):
        super().__init__(); self.capacity = capacity
    def __getitem__(self, key):
        value = super().__getitem__(key); self.move_to_end(key); return value
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.capacity: self.popitem(last=False)

# --- 역할 분리 1: Qdrant 통신 담당 ---
class QdrantManager:
    def __init__(self, client: QdrantClient):
        self.client = client

    def search_batch(self, query_embedding: Dict, config: Dict) -> List[List]:
        conds = [];
        if (page := config.get("page")) is not None: conds.append(FieldCondition(key="page_no", match=MatchValue(value=page)))
        if (el_type := config.get("element_type")) is not None: conds.append(FieldCondition(key="element_type", match=MatchValue(value=el_type)))
        filter_obj = Filter(must=conds) if conds else None
        sparse_vec = SparseVector(indices=[int(k) for k in query_embedding['lexical_weights'].keys()], values=list(query_embedding['lexical_weights'].values()))
        limit_k = config['top_k'] * RETRIEVE_CONFIG['k_multiplier']
        search_params = SearchParams(hnsw_ef=RETRIEVE_CONFIG['search_ef'])
        requests = [
            QueryRequest(query=query_embedding['dense_vecs'].tolist(), using="dense", filter=filter_obj, limit=limit_k, with_payload=True, params=search_params),
            QueryRequest(query=sparse_vec, using="sparse", filter=filter_obj, limit=limit_k, with_payload=True)
        ]
        batch_result = self.client.query_batch_points(collection_name=COLLECTION_NAME, requests=requests)
        dense_hits = batch_result[0].points if batch_result and batch_result[0].points else []
        sparse_hits = batch_result[1].points if len(batch_result) > 1 and batch_result[1].points else []
        logger.info(f"     - 초기 후보군 확보: Dense {len(dense_hits)}개, Sparse {len(sparse_hits)}개")
        return [dense_hits, sparse_hits]

    def get_parents(self, refs: List[str]) -> Dict[str, Any]:
        points, _ = self.client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(should=[FieldCondition(key="self_ref", match=MatchValue(value=r)) for r in refs]),
            limit=len(refs), with_payload=True, with_vectors=True
        )
        return {p.payload.get("self_ref"): p for p in points if p.payload.get("self_ref")}

# --- 역할 분리 2: 재정렬 로직 담당 ---
class Reranker:
    def __init__(self, qdrant_manager: QdrantManager, cache: LRUCache):
        self.qdrant_manager = qdrant_manager
        self.cache = cache
        self.config = RETRIEVE_CONFIG

    def rerank(self, initial_results: List[List], query_embedding: Dict, top_k: int, timings: Dict) -> List[Dict]:
        first = next((r[0] for r in initial_results if r), None)
        is_contextualized = first and first.payload and first.payload.get("is_contextualized", False)
        if is_contextualized:
            logger.info("⚡️ 구조화된 문서 감지. Context-Aware 경로를 실행합니다.")
            return self._context_path(initial_results, query_embedding, top_k, timings)
        else:
            logger.info("⚡️ 비구조화 문서 감지. Simple 가중합 경로를 실행합니다.")
            return self._simple_path(initial_results, top_k)

    def _context_path(self, results: List[List], q_embedding: Dict, top_k: int, timings: Dict) -> List[Dict]:
        with SimpleTimer("  - (Rerank) RRF 융합", timings, log_on_exit=True):
            fused = self._rrf_fusion(results)
        
        scores = self._normalize_combine(fused, ["rrf_score"], self.config['context_path_weights'])
        for i, r in enumerate(fused): r["base_score"] = scores[i]
        fused.sort(key=lambda x: x["base_score"], reverse=True)
        
        with SimpleTimer("  - (Rerank) 부모 문맥 재정렬", timings, log_on_exit=True):
            reranked = self._parent_rerank(fused[:top_k * self.config['k_multiplier']], q_embedding['dense_vecs'])
        
        with SimpleTimer("  - (Rerank) 다양성 필터링", timings, log_on_exit=True):
            final_list = self._ensure_diversity(reranked, top_k)
        return final_list

    def _simple_path(self, results: List[List], top_k: int) -> List[Dict]:
        merged = {};
        for h in results[0]: merged[h.id] = {"point": h, "dense_score": h.score, "sparse_score": 0}
        for h in results[1]:
            if h.id in merged: merged[h.id]["sparse_score"] = h.score
            else: merged[h.id] = {"point": h, "dense_score": 0, "sparse_score": h.score}
        candidates = list(merged.values())
        scores = self._normalize_combine(candidates, ["dense_score", "sparse_score"], self.config['simple_path_weights'])
        for i, c in enumerate(candidates): c["final_score"] = scores[i]
        return sorted(candidates, key=lambda x: x["final_score"], reverse=True)[:top_k]

    def _parent_rerank(self, candidates: List[Dict], q_dense: np.ndarray) -> List[Dict]:
        parent_refs_to_fetch = list(set(c["point"].payload.get("parent_ref") for c in candidates if c["point"].payload and c["point"].payload.get("parent_ref") and c["point"].payload.get("parent_ref") not in self.cache))
        if parent_refs_to_fetch:
            fetched_parents = self.qdrant_manager.get_parents(parent_refs_to_fetch)
            for ref, parent_point in fetched_parents.items(): self.cache[ref] = parent_point
        for c in candidates:
            score = 0.0
            if ref := c["point"].payload.get("parent_ref"):
                if parent := self.cache.get(ref):
                    if parent_vector_attr := getattr(parent, 'vector', None):
                         if dense_vec := parent_vector_attr.get('dense'):
                             score = float(np.dot(q_dense, np.array(dense_vec)))
            c["parent_context_score"] = score
        if any(c["parent_context_score"] > 0 for c in candidates):
            with_parent = sum(1 for c in candidates if c["point"].payload.get("parent_ref")); ratio = with_parent / len(candidates) if candidates else 0
            cfg = self.config['parent_rerank_config']; alpha = cfg["default"]
            for thr, a in cfg["thresholds"]:
                if ratio > thr: alpha = a; break
            logger.info(f"     - 적응형 Alpha: 구조화 비율 {ratio:.1%} -> alpha {alpha} 적용")
            norm_parent = minmax_scale(np.array([c["parent_context_score"] for c in candidates]).reshape(-1, 1)).flatten()
            for i, c in enumerate(candidates): c["final_score"] = alpha * c["base_score"] + (1 - alpha) * norm_parent[i]
        else:
            for c in candidates: c["final_score"] = c["base_score"]
        return sorted(candidates, key=lambda x: x["final_score"], reverse=True)

    def _rrf_fusion(self, results: List[List], k: int = 60) -> List[Dict]:
        scores, points = defaultdict(float), {}; all_hits = [hit for res_list in results for hit in res_list]
        for hit in all_hits: points[hit.id] = hit
        for res_list in results:
            for rank, hit in enumerate(res_list): scores[hit.id] += 1 / (k + rank)
        sorted_ids = sorted(scores, key=scores.get, reverse=True)
        return [{"point": points[id_], "rrf_score": scores[id_]} for id_ in sorted_ids]

    def _ensure_diversity(self, candidates: List[Dict], top_k: int) -> List[Dict]:
        if len(candidates) <= top_k: return candidates
        selected, parent_count = [], defaultdict(int)
        for c in candidates:
            if len(selected) >= top_k: break
            ref = c["point"].payload.get("parent_ref")
            if not ref or parent_count[ref] < self.config['diversity_limit']:
                selected.append(c)
                if ref: parent_count[ref] += 1
        if len(selected) < top_k:
            selected_ids = {s['point'].id for s in selected}; remaining = [c for c in candidates if c['point'].id not in selected_ids]
            selected.extend(remaining[:top_k - len(selected)])
        logger.info(f"     - 다양성 필터링: {len(candidates)}개 후보 -> {len(selected)}개 선택")
        return selected
    
    def _normalize_combine(self, items: List[Dict], keys: List[str], weights: Dict[str, float]) -> np.ndarray:
        if not items: return np.array([])
        combined = np.zeros(len(items))
        for key in keys:
            scores = np.array([item.get(key, 0) for item in items]).reshape(-1, 1)
            norm = minmax_scale(scores).flatten() if np.std(scores) > 0 else np.full(len(items), 0.5)
            combined += weights.get(key.replace("_score", ""), 0) * norm
        return combined

# --- 메인 검색 서비스 클래스 (오케스트레이터) ---
class AdaptiveHybridSearch:
    def __init__(self):
        self.model = BGEM3FlagModel(MODEL_NAME, use_fp16=True)
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=300)
        cache = LRUCache(1000)
        self.qdrant_manager = QdrantManager(qdrant_client)
        self.reranker = Reranker(self.qdrant_manager, cache)
        self.__last_query = ""
        logger.info(f"✅ 최종 개선 버전 검색 시스템 초기화 완료")

    def search(self, query: str, **kwargs) -> List[Dict]:
        self.__last_query = query
        logger.info(f"▶️  새로운 검색 요청: '{query}', {kwargs}")
        
        timings = {} # ✅ 성능 측정용 딕셔너리
        t_total_start = time.monotonic()
        
        with SimpleTimer("쿼리 임베딩", timings, log_on_exit=True):
            query_embedding = self.model.encode(query, return_dense=True, return_sparse=True)
        
        search_config = {"top_k": kwargs.get("top_k", 5), "page": kwargs.get("page"), "element_type": kwargs.get("element_type")}
        with SimpleTimer("초기 DB 검색", timings, log_on_exit=True):
            initial_results = self.qdrant_manager.search_batch(query_embedding, search_config)
        
        if not any(initial_results): return []
        
        with SimpleTimer("재정렬 (전체)", timings): # 이 시간은 세부 단계의 합
            final_points = self.reranker.rerank(initial_results, query_embedding, search_config['top_k'], timings)
        
        with SimpleTimer("결과 포맷팅", timings, log_on_exit=True):
            formatted_results = [self._format_result(r, i) for i, r in enumerate(final_points, 1)]
        
        timings["총 검색 시간"] = (time.monotonic() - t_total_start) * 1000
        
        # 최종 성능 요약 로그 출력
        logger.info("-" * 50)
        logger.info("⏱️  [성능 요약]")
        # 순서를 위해 주요 키를 먼저 정의
        display_order = ["쿼리 임베딩", "초기 DB 검색", "재정렬 (전체)", "  - (Rerank) RRF 융합", "  - (Rerank) 부모 문맥 재정렬", "  - (Rerank) 다양성 필터링", "결과 포맷팅", "총 검색 시간"]
        for key in display_order:
            if key in timings:
                logger.info(f"  - {key:<25}: {timings[key]:.2f} ms")
        logger.info("-" * 50)
        
        return formatted_results

    def _format_result(self, r: Dict, rank: int) -> Dict:
        p = r["point"].payload; final_score = r.get('final_score', r.get('base_score', 0))
        return {"page_content": p.get("text"), "metadata": {"source": p.get("source_file"), "page": p.get("page_no"), "element_type": p.get("element_type"), "relevance_score": round(final_score, 4), "rank": rank, "query": self.__last_query}}

# --- API 서버 ---
searcher = AdaptiveHybridSearch()
mcp = FastMCP("AdaptiveHybridSearchMCP")
@mcp.tool()
async def search_documents(query: str, top_k: int = 5, page_filter: Optional[int] = None, element_type: Optional[str] = None) -> str:
    try:
        docs = searcher.search(query=query, top_k=top_k, page=page_filter, element_type=element_type)
        return json.dumps(docs, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"API 'search_documents' 오류: {e}", exc_info=True)
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    try:
        print("🚀 v0.1 최종 개선 버전 검색 서버 시작")
        print("📍 http://0.0.0.0:8000/sse\n")
        mcp.run(transport="sse", host="0.0.0.0", port=8000, path="/sse")
    except Exception as e:
        print(f"❌ 서버 시작 중 오류 발생: {e}")
        exit(1)
