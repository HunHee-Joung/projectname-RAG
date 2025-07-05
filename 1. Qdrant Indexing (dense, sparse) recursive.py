# index.py
import os
import re
import logging
import subprocess
import tempfile
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

    def _convert_doc_to_pdf(self, file_path: str) -> str:
        """리브레오피스를 사용하여 DOC/DOCX 파일을 PDF로 변환합니다."""
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        
        # DOC/DOCX 파일이 아니면 원본 경로 반환
        if file_ext not in ['.doc', '.docx']:
            return str(file_path)
        
        logger.info(f"📄 DOC 파일 PDF 변환 시작: {file_path.name}")
        
        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp()
        temp_pdf_path = Path(temp_dir) / f"{file_path.stem}.pdf"
        
        try:
            # 리브레오피스를 사용한 PDF 변환 명령어
            cmd = [
                "libreoffice",
                "--headless",  # GUI 없이 실행
                "--convert-to", "pdf",
                "--outdir", temp_dir,
                str(file_path)
            ]
            
            logger.info(f"🔄 리브레오피스 변환 실행: {' '.join(cmd)}")
            
            # 명령어 실행
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5분 타임아웃
                check=True
            )
            
            # 변환된 PDF 파일 확인
            if temp_pdf_path.exists():
                # 원본 파일과 같은 디렉토리에 PDF 파일 복사
                final_pdf_path = file_path.parent / f"{file_path.stem}.pdf"
                
                # 이미 같은 이름의 PDF가 있다면 덮어쓰기 확인
                if final_pdf_path.exists():
                    logger.warning(f"⚠️  기존 PDF 파일이 존재합니다. 덮어씁니다: {final_pdf_path.name}")
                
                # PDF 파일 복사
                import shutil
                shutil.copy2(temp_pdf_path, final_pdf_path)
                
                logger.info(f"✅ PDF 변환 완료: {final_pdf_path.name}")
                return str(final_pdf_path)
            else:
                raise FileNotFoundError(f"변환된 PDF 파일을 찾을 수 없습니다: {temp_pdf_path}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ PDF 변환 타임아웃: {file_path.name}")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ PDF 변환 실패: {file_path.name}")
            logger.error(f"   오류 코드: {e.returncode}")
            logger.error(f"   표준 출력: {e.stdout}")
            logger.error(f"   표준 에러: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"❌ PDF 변환 중 예상치 못한 오류: {e}")
            raise
        finally:
            # 임시 파일 정리
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.debug(f"🧹 임시 디렉토리 정리 완료: {temp_dir}")
            except Exception as e:
                logger.warning(f"⚠️  임시 디렉토리 정리 실패: {e}")

    def store_document(self, file_path: str) -> int:
        """지정된 파일을 로드하고 Qdrant에 인덱싱하는 전체 프로세스를 실행합니다."""
        logger.info(f"📁 문서 로딩 및 인덱싱 시작: {file_path}")
        
        # 0. DOC 파일인 경우 PDF로 변환
        try:
            processed_file_path = self._convert_doc_to_pdf(file_path)
            if processed_file_path != file_path:
                logger.info(f"🔄 변환된 파일 사용: {processed_file_path}")
        except Exception as e:
            logger.error(f"❌ DOC 파일 변환 실패: {e}")
            logger.info("📄 원본 파일로 계속 진행합니다...")
            processed_file_path = file_path
        
        # 1. 문서 로드 및 청킹
        loader = DoclingLoader(file_path=processed_file_path, chunker=HybridChunker(tokenizer=MODEL_NAME, merge_peers=True, max_context_length=self.config["max_context_length"], contextualize=True))
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
        
        # 5. 임베딩 및 업로드 (시작 ID부터, 원본 파일 경로로 저장)
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

def process_directory(engine: HybridSearchEngine, directory_path: str) -> Dict[str, Any]:
    """지정된 디렉토리의 모든 지원 파일을 재귀적으로 처리합니다."""
    
    # 지원하는 파일 확장자
    SUPPORTED_EXTENSIONS = {'.pdf', '.doc', '.docx'}
    
    directory = Path(directory_path)
    if not directory.exists():
        logger.error(f"❌ 디렉토리를 찾을 수 없습니다: {directory_path}")
        return {"status": "error", "message": "Directory not found"}
    
    if not directory.is_dir():
        logger.error(f"❌ 경로가 디렉토리가 아닙니다: {directory_path}")
        return {"status": "error", "message": "Path is not a directory"}
    
    # 지원되는 모든 파일 찾기 (재귀적)
    all_files = []
    for ext in SUPPORTED_EXTENSIONS:
        # **/* 패턴으로 모든 하위 디렉토리까지 검색
        files = list(directory.rglob(f"*{ext}"))
        all_files.extend(files)
    
    if not all_files:
        logger.warning(f"⚠️  지원되는 파일을 찾을 수 없습니다. 지원 형식: {', '.join(SUPPORTED_EXTENSIONS)}")
        return {"status": "warning", "message": "No supported files found", "processed": 0, "errors": 0}
    
    logger.info(f"📂 총 {len(all_files)}개의 파일을 발견했습니다.")
    logger.info(f"🎯 처리 대상 디렉토리: {directory_path}")
    logger.info("-" * 80)
    
    # 처리 결과 추적
    results = {
        "total_files": len(all_files),
        "processed_files": 0,
        "error_files": 0,
        "total_points": 0,
        "success_files": [],
        "error_details": [],
        "start_time": datetime.now()
    }
    
    # 각 파일 처리
    for idx, file_path in enumerate(all_files, 1):
        relative_path = file_path.relative_to(directory)
        logger.info(f"\n📄 [{idx}/{len(all_files)}] 처리 중: {relative_path}")
        
        try:
            # 파일 크기 정보 추가
            file_size = file_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            logger.info(f"   📐 파일 크기: {file_size_mb:.2f} MB")
            
            # 문서 처리
            points_count = engine.store_document(str(file_path))
            
            # 성공 기록
            results["processed_files"] += 1
            results["total_points"] += points_count
            results["success_files"].append({
                "file": str(relative_path),
                "points": points_count,
                "size_mb": round(file_size_mb, 2)
            })
            
            logger.info(f"   ✅ 완료: {points_count}개 포인트 추가")
            
        except Exception as e:
            # 에러 기록
            results["error_files"] += 1
            error_info = {
                "file": str(relative_path),
                "error": str(e),
                "error_type": type(e).__name__
            }
            results["error_details"].append(error_info)
            
            logger.error(f"   ❌ 실패: {relative_path}")
            logger.error(f"      오류: {e}")
            
            # 에러가 발생해도 다음 파일 계속 처리
            continue
    
    # 처리 완료 시간 기록
    results["end_time"] = datetime.now()
    results["duration"] = (results["end_time"] - results["start_time"]).total_seconds()
    
    return results

def print_processing_summary(results: Dict[str, Any]):
    """처리 결과 요약을 출력합니다."""
    logger.info("\n" + "="*30 + " 처리 완료 요약 " + "="*30)
    logger.info(f"📊 전체 파일: {results['total_files']}개")
    logger.info(f"✅ 성공: {results['processed_files']}개")
    logger.info(f"❌ 실패: {results['error_files']}개")
    logger.info(f"📈 총 생성된 포인트: {results['total_points']:,}개")
    logger.info(f"⏱️  총 처리 시간: {results['duration']:.1f}초")
    
    if results['success_files']:
        logger.info(f"\n✅ 성공한 파일들:")
        for file_info in results['success_files']:
            logger.info(f"   📄 {file_info['file']} → {file_info['points']}개 포인트 ({file_info['size_mb']}MB)")
    
    if results['error_details']:
        logger.info(f"\n❌ 실패한 파일들:")
        for error_info in results['error_details']:
            logger.info(f"   📄 {error_info['file']}")
            logger.info(f"      🔸 오류 유형: {error_info['error_type']}")
            logger.info(f"      🔸 오류 메시지: {error_info['error']}")
    
    # 성공률 계산
    if results['total_files'] > 0:
        success_rate = (results['processed_files'] / results['total_files']) * 100
        logger.info(f"\n📈 성공률: {success_rate:.1f}%")
    
    logger.info("="*75 + "\n")

if __name__ == "__main__":
    engine = HybridSearchEngine()
    data_directory = "data"
    
    logger.info(f"🚀 디렉토리 처리 시작: {data_directory}")
    
    try:
        # 디렉토리 전체 처리
        results = process_directory(engine, data_directory)
        
        # 결과 요약 출력
        if results.get("status") == "error":
            logger.error(f"❌ 처리 실패: {results.get('message', 'Unknown error')}")
        elif results.get("status") == "warning":
            logger.warning(f"⚠️  {results.get('message', 'Unknown warning')}")
        else:
            print_processing_summary(results)
            
            if results['processed_files'] > 0:
                logger.info(f"🎉 배치 처리 성공! 총 {results['processed_files']}개 파일에서 {results['total_points']:,}개의 포인트가 생성되었습니다.")
            else:
                logger.warning("⚠️  처리된 파일이 없습니다.")
                
    except KeyboardInterrupt:
        logger.warning("⚠️  사용자에 의해 처리가 중단되었습니다.")
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류 발생: {e}", exc_info=True)
