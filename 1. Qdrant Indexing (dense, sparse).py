# index.py
import os
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict, Counter

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import *
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
from FlagEmbedding import BGEM3FlagModel

# --- Qdrant 및 모델 공통 설정 ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "192.168.0.249")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docling_search")
MODEL_NAME = "BAAI/bge-m3"


# --- index.py 전용 설정 ---
INDEX_CONFIG = {
    "batch_size": 10,
    "max_context_length": 8192,
    "group_text_limit": 500,
    "group_total_limit": 2000,
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridSearchEngine:
    """문서 처리, 임베딩, Qdrant 인덱싱을 담당하는 클래스"""
    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=300)
        self.model = None
        self.config = INDEX_CONFIG
        logger.info(f"✅ 검색 엔진 초기화: {QDRANT_HOST}:{QDRANT_PORT}")
    
    def _get_model(self) -> BGEM3FlagModel:
        """필요할 때 BGE-M3 모델을 한 번만 로드합니다."""
        if not self.model:
            logger.info("🤖 BGE-M3 모델 로딩...")
            self.model = BGEM3FlagModel(MODEL_NAME, use_fp16=True)
        return self.model

    # def store_document(self, file_path: str) -> int:
    #     """지정된 파일을 로드하고 Qdrant에 인덱싱하는 전체 프로세스를 실행합니다."""
    #     logger.info(f"📁 문서 로딩 및 인덱싱 시작: {file_path}")
        
    #     # 1. 문서 로드 및 청킹
    #     loader = DoclingLoader(file_path=file_path, chunker=HybridChunker(tokenizer=MODEL_NAME, merge_peers=True, max_context_length=self.config["max_context_length"], contextualize=True))
    #     chunks = loader.load()
    #     logger.info(f"📄 원본 청크 {len(chunks)}개 로드 완료")
        
    #     # 2. 기존 컬렉션 삭제
    #     if self.client.collection_exists(collection_name=COLLECTION_NAME):
    #         logger.info(f"🗑️ 기존 컬렉션 '{COLLECTION_NAME}' 삭제...")
    #         self.client.delete_collection(collection_name=COLLECTION_NAME)

    #     # 3. 새 컬렉션 생성 (최적화된 설정 적용)
    #     self.client.create_collection(
    #         collection_name=COLLECTION_NAME,
    #         vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE, hnsw_config=HnswConfigDiff(m=16, ef_construct=100))},
    #         sparse_vectors_config={"sparse": SparseVectorParams()}
    #     )
    #     logger.info(f"✨ 새 컬렉션 생성 완료 (HNSW 최적화 적용)")
        
    #     # 4. 페이로드 인덱스 생성
    #     self._create_payload_indexes()
        
    #     # 5. Group 청크 생성 및 전체 청크 병합
    #     group_chunks = self._create_group_chunks(chunks)
    #     all_chunks = chunks + group_chunks
        
    #     # 6. 임베딩 및 업로드
    #     self._embed_and_upload_chunks(all_chunks, file_path)
        
    #     # 7. 결과 분석 및 보고
    #     stats = self._collect_stats(chunks)
    #     self._log_results(stats, len(chunks), len(group_chunks))
        
    #     return len(all_chunks)

    def store_document(self, file_path: str) -> int:
        """지정된 파일을 로드하고 Qdrant에 인덱싱하는 전체 프로세스를 실행합니다."""
        logger.info(f"📁 문서 로딩 및 인덱싱 시작: {file_path}")
        
        # 1. 문서 로드 및 청킹
        loader = DoclingLoader(file_path=file_path, chunker=HybridChunker(tokenizer=MODEL_NAME, merge_peers=True, max_context_length=self.config["max_context_length"], contextualize=True))
        chunks = loader.load()
        logger.info(f"📄 원본 청크 {len(chunks)}개 로드 완료")
        
        # 2. 컬렉션 존재 여부 확인 및 생성
        if not self.client.collection_exists(collection_name=COLLECTION_NAME):
            logger.info(f"✨ 새 컬렉션 '{COLLECTION_NAME}' 생성 중...")
            # 새 컬렉션 생성 (최적화된 설정 적용)
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE, hnsw_config=HnswConfigDiff(m=16, ef_construct=100))},
                sparse_vectors_config={"sparse": SparseVectorParams()}
            )
            logger.info(f"✅ 새 컬렉션 생성 완료 (HNSW 최적화 적용)")
            
            # 페이로드 인덱스 생성 (새 컬렉션인 경우에만)
            self._create_payload_indexes()
        else:
            logger.info(f"📦 기존 컬렉션 '{COLLECTION_NAME}' 사용")
        
        # 3. 시작 ID 계산 (기존 포인트 개수 기반)
        collection_info = self.client.get_collection(collection_name=COLLECTION_NAME)
        start_id = collection_info.points_count
        logger.info(f"🔢 시작 ID: {start_id} (기존 포인트: {collection_info.points_count}개)")
        
        # 4. Group 청크 생성 및 전체 청크 병합
        group_chunks = self._create_group_chunks(chunks)
        all_chunks = chunks + group_chunks
        
        # 5. 임베딩 및 업로드 (시작 ID부터)
        self._embed_and_upload_chunks(all_chunks, file_path, start_id)
        
        # 6. 결과 분석 및 보고
        stats = self._collect_stats(chunks)
        self._log_results(stats, len(chunks), len(group_chunks))
        
        return len(all_chunks)
    def _create_payload_indexes(self):
        """필터링에 사용할 필드에 대한 페이로드 인덱스를 생성합니다."""
        payload_fields = {"self_ref": PayloadSchemaType.KEYWORD, "page_no": PayloadSchemaType.INTEGER, "element_type": PayloadSchemaType.KEYWORD}
        for field, schema_type in payload_fields.items():
            logger.info(f"⚡️ '{field}' 필드에 Payload Index 생성 중...")
            self.client.create_payload_index(collection_name=COLLECTION_NAME, field_name=field, field_schema=schema_type, wait=True)
        logger.info("✅ 모든 Payload Index 생성 완료!")
        
    # def _embed_and_upload_chunks(self, chunks: List[Any], file_path: str):
    #     """청크를 배치 단위로 임베딩하고 Qdrant에 업로드합니다."""
    #     model = self._get_model()
    #     logger.info(f"🚀 총 {len(chunks)}개 청크 임베딩 및 업로드 시작 (배치 크기: {self.config['batch_size']})...")
        
    #     for i in range(0, len(chunks), self.config["batch_size"]):
    #         batch_chunks = chunks[i:i + self.config["batch_size"]]
    #         texts_to_embed = [chunk.page_content for chunk in batch_chunks]

    #         batch_outputs = model.encode(texts_to_embed, return_dense=True, return_sparse=True)
            
    #         points = []
    #         for j, chunk in enumerate(batch_chunks):
    #             meta = self._extract_metadata(chunk)
    #             points.append(PointStruct(
    #                 id=i + j,
    #                 vector={
    #                     "dense": batch_outputs["dense_vecs"][j].tolist(), 
    #                     "sparse": SparseVector(indices=[int(k) for k in batch_outputs["lexical_weights"][j].keys()], values=list(batch_outputs["lexical_weights"][j].values()))
    #                 },
    #                 payload={"text": chunk.page_content, "source_file": file_path, **meta} # ✅ ColBERT 페이로드 없음
    #             ))
    #         self.client.upsert(COLLECTION_NAME, points, wait=True)
    #         logger.info(f"     ... 진행: {i + len(batch_chunks)}/{len(chunks)}개 청크 업로드 완료.")
    
    def _embed_and_upload_chunks(self, chunks: List[Any], file_path: str, start_id: int = 0):
        """청크를 배치 단위로 임베딩하고 Qdrant에 업로드합니다."""
        model = self._get_model()
        logger.info(f"🚀 총 {len(chunks)}개 청크 임베딩 및 업로드 시작 (배치 크기: {self.config['batch_size']}, 시작 ID: {start_id})...")
        
        for i in range(0, len(chunks), self.config["batch_size"]):
            batch_chunks = chunks[i:i + self.config["batch_size"]]
            texts_to_embed = [chunk.page_content for chunk in batch_chunks]

            batch_outputs = model.encode(texts_to_embed, return_dense=True, return_sparse=True)
            
            points = []
            for j, chunk in enumerate(batch_chunks):
                meta = self._extract_metadata(chunk)
                points.append(PointStruct(
                    id=start_id + i + j,  # 시작 ID부터 증가
                    vector={
                        "dense": batch_outputs["dense_vecs"][j].tolist(), 
                        "sparse": SparseVector(indices=[int(k) for k in batch_outputs["lexical_weights"][j].keys()], values=list(batch_outputs["lexical_weights"][j].values()))
                    },
                    payload={"text": chunk.page_content, "source_file": file_path, **meta}
                ))
            self.client.upsert(COLLECTION_NAME, points, wait=True)
            logger.info(f"     ... 진행: {i + len(batch_chunks)}/{len(chunks)}개 청크 업로드 완료.")

    def _extract_metadata(self, chunk: Any) -> Dict[str, Any]:
        meta = chunk.metadata; dl_meta = meta.get('dl_meta', {}); doc_items = dl_meta.get('doc_items', [])
        result = {'page_no': None, 'bbox': None, 'element_type': 'unknown', 'headings': dl_meta.get('headings', []), 'parent_ref': None, 'self_ref': None, 'children_refs': [], 'structure_labels': [], 'is_contextualized': False}
        if not doc_items: return result
        item = doc_items[0] if isinstance(doc_items, list) else doc_items; prov = item.get('prov', [{}])[0]
        result.update({'page_no': prov.get('page_no'), 'bbox': prov.get('bbox'), 'self_ref': item.get('self_ref'),'children_refs': [c['$ref'] for c in item.get('children', []) if isinstance(c, dict) and '$ref' in c]})
        text_label = f"{item.get('self_ref', '')} {item.get('label', '')}".lower()
        if 'table' in text_label: result['element_type'] = 'table'
        elif 'figure' in text_label: result['element_type'] = 'figure'
        elif any(x in text_label for x in ['heading', 'title']): result['element_type'] = 'heading'
        else: result['element_type'] = 'text'
        if parent := item.get('parent', {}):
            if '$ref' in parent: result['parent_ref'] = parent['$ref']
        if label := item.get('label'): result['structure_labels'].append(label)
        result['is_contextualized'] = self._is_contextualized(result, chunk.page_content)
        return result
    
    def _is_contextualized(self, meta: Dict[str, Any], text: str) -> bool:
        score = ((0.4 if meta['headings'] else 0) + (0.3 if meta['parent_ref'] and meta['parent_ref'] != '#/body' else 0) + (0.2 if any(label in {'section_header','title','table','figure','heading','caption','list_item'} for label in meta['structure_labels']) else 0) + (0.1 if meta['element_type'] in {'table','figure','heading','title'} else 0))
        if len(text) <= 200 and any(re.match(p, text.strip()) for p in [r'^\d+\.\s', r'^[A-Z][A-Z\s]{2,}$', r'^Chapter\s+\d+', r'^Section\s+\d+']): score += 0.1
        return score >= 0.3
    
    def _create_group_chunks(self, chunks: List[Any]) -> List[Any]:
        groups = defaultdict(lambda: {'texts': [], 'page_nos': []})
        for chunk in chunks:
            meta = self._extract_metadata(chunk)
            if (ref := meta['parent_ref']) and ref.startswith('#/groups/'):
                groups[ref]['texts'].append(chunk.page_content[:self.config["group_text_limit"]]);
                if meta['page_no'] is not None: groups[ref]['page_nos'].append(meta['page_no'])
        group_chunks = []
        for ref, data in groups.items():
            text = " | ".join(data['texts'])
            if len(text) > self.config["group_total_limit"]: text = text[:self.config["group_total_limit"]] + "..."
            page_no = min(data['page_nos']) if data['page_nos'] else None
            chunk_obj = type('obj', (object,), {'page_content': f"Group Summary: {text}",'metadata': {'dl_meta': {'doc_items': [{'self_ref': ref, 'parent': {'$ref': '#/body'}, 'label': 'group', 'prov': [{'page_no': page_no}]}]}}})()
            group_chunks.append(chunk_obj)
        return group_chunks

    def _collect_stats(self, chunks: List[Any]) -> Dict[str, Any]:
        stats = {'contextualized': 0, 'with_headings': 0, 'with_parents': 0, 'types': Counter()}
        for chunk in chunks:
            meta = self._extract_metadata(chunk)
            if meta['is_contextualized']: stats['contextualized'] += 1
            if meta['headings']: stats['with_headings'] += 1
            if meta['parent_ref']: stats['with_parents'] += 1
            stats['types'][meta['element_type']] += 1
        return stats
    
    def _log_results(self, stats: Dict[str, Any], original: int, groups: int):
        if original == 0: logger.warning("처리할 원본 청크가 없어 통계를 생략합니다."); return
        total = original + groups; ctx_ratio = stats['contextualized'] / original if original > 0 else 0
        logger.info("\n" + "="*25 + " 인덱싱 분석 결과 " + "="*25); logger.info(f"  - 원본 청크: {original}개 | Group 청크: {groups}개 | 총 인덱싱: {total}개"); logger.info("-" * 65); logger.info(f"  - 구조화된 청크 비율: {stats['contextualized']}/{original} ({ctx_ratio:.1%})"); logger.info(f"  - 요소 타입 분포: {dict(stats['types'])}")
        if ctx_ratio > 0.5: recommendation = "Context-Aware Path"
        elif ctx_ratio > 0.2: recommendation = "Hybrid Path"
        else: recommendation = "Simple Path"
        logger.info(f"  - 검색 경로 추천: ✅ {recommendation}"); logger.info("="*65 + "\n")

if __name__ == "__main__":
    engine = HybridSearchEngine()
    sample_file = "data/sample_document.pdf"
    if Path(sample_file).exists():
        try:
            count = engine.store_document(sample_file)
            logger.info(f"🎉 인덱싱 작업 성공. 총 {count}개의 포인트가 저장되었습니다.")
        except Exception as e:
            logger.error(f"❌ 인덱싱 작업 중 심각한 오류 발생: {e}", exc_info=True)
    else:
        logger.warning(f"'{sample_file}' 파일을 찾을 수 없습니다.")
