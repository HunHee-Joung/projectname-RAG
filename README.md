## 개요 (Overview)

이 노트북은 **지능형 하이브리드 검색(Intelligent Hybrid Search) 시스템**을 구축하고 실행하는 전체 과정을 담고 있습니다. PDF와 같은 문서를 입력받아, 의미적(Semantic) 검색과 키워드 기반(Lexical) 검색을 결합한 하이브리드 방식으로 검색할 수 있도록 인덱싱하고, 문서의 구조적 특징을 분석하여 검색 시 동적으로 최적의 재정렬(Reranking) 전략을 적용하는 **적응형 검색(Adaptive Search) API 서버**를 제공합니다. 마지막으로, 인덱싱된 데이터를 쉽게 확인할 수 있는 유틸리티 스크립트도 포함되어 있습니다.

-----

## 사전 준비 (Prerequisites)

이 시스템을 실행하기 위해 사용되는 주요 라이브러리 및 프레임워크는 다음과 같습니다.

  * **데이터 처리 및 임베딩:**
      * `langchain_docling`, `docling`: 문서를 구조적으로 분석하고 청킹(Chunking)하기 위한 라이브러리입니다.
      * `FlagEmbedding`: `BAAI/bge-m3` 모델을 사용하여 텍스트로부터 Dense Vector(의미)와 Sparse Vector(키워드) 임베딩을 동시에 생성합니다.
      * `numpy`: 수치 연산 및 벡터 처리를 위해 사용됩니다.
      * `pandas`: 인덱싱된 데이터를 표 형태로 확인하기 위해 사용됩니다.
  * **벡터 데이터베이스:**
      * `qdrant_client`: 고성능 벡터 데이터베이스인 Qdrant와 통신하기 위한 클라이언트입니다.
  * **API 서버 및 기타:**
      * `fastmcp`: 검색 기능을 API로 노출하기 위한 경량 비동기 서버 프레임워크입니다.
      * `sklearn`: 검색 결과의 점수를 정규화(min-max scaling)하는 데 사용됩니다.
      * `logging`, `os`, `time`, `re`: 표준 라이브러리로, 로깅, 환경 변수 관리, 시간 측정, 정규 표현식 등에 사용됩니다.

-----

## 상세 분석 및 설명

### 1\. Qdrant 인덱싱 (`1. Qdrant Indexing (dense, sparse).py`)

이 스크립트는 원본 문서를 검색 가능한 형태로 가공하여 Qdrant 벡터 데이터베이스에 저장하는 역할을 담당합니다.

  * **What (무엇을):**
    `HybridSearchEngine` 클래스는 지정된 PDF 파일을 `DoclingLoader`로 로드하여 구조적 정보를 포함한 청크로 분할합니다. 이후 `BGE-M3` 모델을 사용해 각 청크에서 Dense 벡터와 Sparse 벡터를 추출하고, 텍스트 및 메타데이터와 함께 Qdrant에 저장(인덱싱)합니다.

  * **Why (왜):**
    단순 텍스트 분할을 넘어, `HybridChunker`를 통해 문서의 제목, 표, 리스트 같은 구조적 요소를 최대한 보존하며 청킹합니다. 이는 이후 검색 단계에서 문맥을 더 잘 이해하는 기반이 됩니다. 또한 Dense(의미)와 Sparse(키워드) 벡터를 모두 사용하는 하이브리드 방식은 검색 정확도를 높여줍니다. 기존 컬렉션을 삭제하고 새로 생성하는 과정을 통해 항상 최신 상태의 깨끗한 인덱스를 유지합니다.

  * **How (어떻게):**

    1.  **초기화 및 모델 로딩**: Qdrant 클라이언트를 초기화하고, 임베딩을 위한 `BGE-M3` 모델을 필요 시점에 한 번만 로드(Lazy Loading)하여 효율성을 높입니다.
    2.  **컬렉션 생성**: `dense`와 `sparse` 벡터를 모두 저장할 수 있도록 Qdrant 컬렉션을 설정합니다. 코사인 유사도와 HNSW 설정을 통해 빠르고 정확한 벡터 검색 환경을 구성합니다.
    3.  **그룹 청크 생성 (`_create_group_chunks`)**: 원본 청크 외에, 논리적으로 연결된 청크들(예: 하나의 섹션에 속한 여러 문단)을 요약하는 '그룹 청크'를 추가로 생성합니다. 이는 특정 섹션 전체에 대한 검색을 가능하게 합니다.
    4.  **임베딩 및 업로드 (`_embed_and_upload_chunks`)**: 모든 청크(원본 + 그룹)를 배치 단위로 처리합니다. `model.encode`를 호출하여 Dense/Sparse 벡터를 얻고, 이를 `PointStruct` 형태로 구성하여 Qdrant에 `upsert`합니다.
    5.  **메타데이터 추출 및 분석 (`_extract_metadata`, `_is_contextualized`)**: 각 청크의 페이지 번호, 요소 타입(표, 제목 등), 부모-자식 관계 등의 구조적 메타데이터를 추출합니다. 이 정보를 바탕으로 해당 청크가 얼마나 '구조화'되었는지 점수를 매겨 `is_contextualized` 플래그를 설정합니다. 이 플래그는 검색 시 적응형 로직을 선택하는 핵심 기준이 됩니다.

#### 코드 블록: `store_document` 핵심 로직

```python
# index.py

def store_document(self, file_path: str) -> int:
    """지정된 파일을 로드하고 Qdrant에 인덱싱하는 전체 프로세스를 실행합니다."""
    logger.info(f"📁 문서 로딩 및 인덱싱 시작: {file_path}")
    
    # 1. 문서 로드 및 청킹
    loader = DoclingLoader(file_path=file_path, chunker=HybridChunker(tokenizer=MODEL_NAME, merge_peers=True, max_context_length=self.config["max_context_length"], contextualize=True))
    chunks = loader.load()
    
    # 2. 기존 컬렉션 삭제 후 새로 생성
    if self.client.collection_exists(collection_name=COLLECTION_NAME):
        self.client.delete_collection(collection_name=COLLECTION_NAME)

    self.client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE, hnsw_config=HnswConfigDiff(m=16, ef_construct=100))},
        sparse_vectors_config={"sparse": SparseVectorParams()}
    )
    
    # 3. 페이로드 인덱스 생성 (빠른 필터링용)
    self._create_payload_indexes()
    
    # 4. Group 청크 생성 및 전체 청크 병합
    group_chunks = self._create_group_chunks(chunks)
    all_chunks = chunks + group_chunks
    
    # 5. 임베딩 및 업로드
    self._embed_and_upload_chunks(all_chunks, file_path)
    
    # 6. 결과 분석 및 보고
    stats = self._collect_stats(chunks)
    self._log_results(stats, len(chunks), len(group_chunks))
    
    return len(all_chunks)
```

-----

### 2\. 적응형 검색 서버 (`2. MCP Server (dense, sparse, rrf).py`)

이 스크립트는 사용자의 쿼리를 받아 Qdrant에서 최적의 문서를 검색하고, 그 결과를 지능적으로 재정렬하여 반환하는 API 서버를 구현합니다.

  * **What (무엇을):**
    `AdaptiveHybridSearch` 클래스는 사용자의 텍스트 쿼리를 임베딩한 후, Qdrant에서 Dense/Sparse 벡터 검색을 동시에 실행하여 1차 후보군을 가져옵니다. 그 후, 검색된 문서의 특성(`is_contextualized`)에 따라 'Context-Aware 경로' 또는 'Simple 경로'라는 두 가지 다른 재정렬 전략 중 하나를 동적으로 선택하여 최종 결과를 반환합니다.

  * **Why (왜):**
    모든 문서나 쿼리에 단일한 검색 방식을 적용하는 것은 비효율적입니다. 인덱싱 단계에서 파악된 문서의 **구조화 정도**에 따라 검색 전략을 달리하는 **적응형 방식**을 사용합니다. 구조화가 잘 된 문서의 경우, 부모-자식 관계 같은 풍부한 문맥 정보를 활용하여 정확도를 극대화하고, 그렇지 않은 일반 문서의 경우엔 안정적인 가중치 기반 하이브리드 모델을 사용하기 위함입니다. `LRUCache`를 사용해 부모 청크 정보를 캐싱하여 반복적인 DB 조회를 피해 성능을 높입니다.

  * **How (어떻게):**

    1.  **역할 분리**: 코드를 `QdrantManager`(DB 통신), `Reranker`(재정렬 로직), `AdaptiveHybridSearch`(전체 흐름 제어) 클래스로 분리하여 유지보수성을 높였습니다.
    2.  **초기 검색 (`QdrantManager.search_batch`)**: 쿼리를 Dense/Sparse 벡터로 변환 후, Qdrant에 `query_batch_points`를 사용하여 두 종류의 검색을 한 번의 요청으로 효율적으로 수행합니다.
    3.  **경로 선택 (`Reranker.rerank`)**: 1차 검색 결과의 최상위 문서가 `is_contextualized` 플래그를 가지고 있는지 확인하여 실행 경로를 결정합니다.
    4.  **Context-Aware 경로 (`_context_path`)**:
          * **RRF 융합 (`_rrf_fusion`)**: Dense와 Sparse 검색 결과를 Reciprocal Rank Fusion(RRF) 알고리즘으로 융합합니다. 이는 별도의 가중치 튜닝 없이도 안정적으로 두 결과를 합치는 기법입니다.
          * **부모 문맥 재정렬 (`_parent_rerank`)**: 각 청크의 **부모 청크**를 DB에서 조회(캐시 우선 확인)하고, 부모 청크와 쿼리의 유사도를 계산합니다. 이 '부모 문맥 점수'를 원래 점수와 결합하여, 좋은 문맥에 속한 청크의 순위를 높여줍니다. (예: 질문과 관련된 섹션 제목 아래에 있는 문단에 가산점 부여)
          * **다양성 필터링 (`_ensure_diversity`)**: 최종 결과에 동일한 부모를 가진 청크가 너무 많이 포함되지 않도록 조절하여, 사용자에게 더 다양한 정보를 제공합니다.
    5.  **Simple 경로 (`_simple_path`)**:
          * Dense 점수와 Sparse 점수를 각각 정규화(0\~1 스케일)한 후, 미리 정의된 가중치(`dense: 0.6, sparse: 0.4`)로 합산하여 최종 점수를 계산합니다. 이는 구조 정보가 부족할 때 안정적인 성능을 내는 표준적인 하이브리드 검색 방식입니다.

#### 코드 블록: `Reranker`의 적응형 경로 선택 로직

```python
# retrieve.py

def rerank(self, initial_results: List[List], query_embedding: Dict, top_k: int, timings: Dict) -> List[Dict]:
    # 1차 검색 결과의 첫 번째 아이템을 확인
    first = next((r[0] for r in initial_results if r), None)
    is_contextualized = first and first.payload and first.payload.get("is_contextualized", False)
    
    # is_contextualized 값에 따라 경로 분기
    if is_contextualized:
        logger.info("⚡️ 구조화된 문서 감지. Context-Aware 경로를 실행합니다.")
        return self._context_path(initial_results, query_embedding, top_k, timings)
    else:
        logger.info("⚡️ 비구조화 문서 감지. Simple 가중합 경로를 실행합니다.")
        return self._simple_path(initial_results, top_k)
```

-----

### 3\. 청킹 문서 정보 확인 (`viewer` 스크립트)

이 스크립트는 인덱싱이 완료된 후, Qdrant 데이터베이스에 저장된 청크들의 정보를 개발자가 직접 눈으로 확인할 수 있게 해주는 유틸리티입니다.

  * **What (무엇을):**
    `QdrantDataViewer` 클래스는 `docling_search` 컬렉션에 접속하여 저장된 모든 데이터 포인트(청크)를 가져옵니다. 각 포인트의 ID, 요소 타입, 페이지 번호, `is_contextualized` 여부, 부모/자식 ID, 그리고 텍스트 미리보기 등 핵심 정보를 `pandas` DataFrame으로 정리하여 터미널에 표 형태로 출력합니다.

  * **Why (왜):**
    복잡한 인덱싱 과정이 의도대로 잘 수행되었는지 검증하기 위해 필수적인 도구입니다. 예를 들어, `parent_ref`가 올바르게 설정되었는지, `element_type`이 정확하게 분류되었는지, `is_contextualized` 플래그가 적절히 부여되었는지 등을 한눈에 파악할 수 있어 디버깅과 시스템 튜닝에 매우 유용합니다.

  * **How (어떻게):**
    `qdrant_client.scroll` API를 사용하여 컬렉션의 모든 데이터를 효율적으로 순회합니다. 각 포인트의 `payload`에서 필요한 정보들을 추출하여 리스트에 담은 후, 이를 `pandas.DataFrame`으로 변환하여 가독성 높은 표로 출력합니다.

#### 코드 블록: `display_all_chunks_summary` 데이터 요약 로직

```python
# viewer 스크립트 (일부)

def display_all_chunks_summary(self):
    """컬렉션의 모든 청크에 대한 요약 정보를 표로 출력합니다."""
    # ... (DB 접속 및 포인트 개수 확인) ...

    # scroll API로 모든 포인트 가져오기
    all_points, _ = self.client.scroll(
        collection_name=COLLECTION_NAME,
        limit=total_points,
        with_payload=True,
        with_vectors=False # 벡터 데이터는 필요 없으므로 제외
    )
    
    summary_data = []
    for point in all_points:
        payload = point.payload
        summary_data.append({
            "ID": point.id,
            "Type": payload.get("element_type"),
            "Page": payload.get("page_no"),
            "Is_Ctx": payload.get("is_contextualized"),
            "Parent_ID": payload.get("parent_ref"),
            "Self_ID": payload.get("self_ref"),
            "Text_Preview": (payload.get("text", "")[:70] + "...")
        })
    
    # Pandas DataFrame으로 변환하여 출력
    df = pd.DataFrame(summary_data)
    # ... (Pandas 출력 옵션 설정) ...
    print(df)
```

-----

## 결론 및 요약

이 노트북은 문서의 구조적 특징을 지능적으로 활용하는 **차세대 적응형 하이브리드 검색 시스템**의 완전한 구현체입니다.

  * **인덱싱 단계**에서는 `docling`과 `BGE-M3` 모델을 통해 단순 텍스트를 넘어선, 문맥 정보가 풍부한(context-rich) 벡터 인덱스를 구축합니다.
  * **검색 단계**에서는 문서의 특성에 따라 RRF, 부모 문맥 재정렬, 다양성 필터링을 포함한 **'Context-Aware' 고급 전략**과 안정적인 **'Simple' 가중합 전략** 사이를 동적으로 전환합니다.
  * 이러한 접근 방식을 통해, 시스템은 어떤 종류의 문서가 주어지더라도 그에 맞는 최적의 검색 품질을 제공할 수 있는 유연성과 성능을 모두 확보했습니다.

결론적으로, 이 노트북은 최신 벡터 검색 기술과 문서 이해(Document Understanding) 기술을 결합하여, 정교하고 효율적인 정보 검색 솔루션을 어떻게 설계하고 구축할 수 있는지 보여주는 훌륭한 청사진입니다.
