# Docling을 사용한 PDF 파일 처리 및 이미지 설명 생성 가이드

PDF 문서에서 텍스트 청킹, 표 추출, 이미지 추출 및 AI를 활용한 이미지 설명 생성을 지원하는 Python 라이브러리입니다.

## 주요 기능

- 📄 PDF 문서 텍스트 청킹
- 📊 표(Table) 자동 추출 및 구조화
- 🖼️ 이미지 추출 및 Base64 인코딩
- 🤖 Ollama qwen2.5vl:7b 모델을 활용한 이미지 설명 생성
- 💾 추출된 데이터를 다양한 형식으로 저장

## 시스템 요구사항

- Python 3.8 이상
- Ollama 서버 (이미지 설명 기능 사용 시)

## 설치

### 1. Python 패키지 설치

```bash
pip install docling ollama pillow
```

### 2. Ollama 설치 및 모델 다운로드

```bash
# Ollama 설치 (macOS)
brew install ollama

# Ollama 설치 (Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Ollama 서버 실행
ollama serve

# 비전 모델 다운로드 (새 터미널에서)
ollama pull qwen2.5vl:7b
```

## 사용법

### 기본 사용 예시

```python
from pdf_processor import PDFProcessor

# 프로세서 초기화
processor = PDFProcessor(
    enable_image_extraction=True,
    enable_ollama_description=True,
    ollama_model="qwen2.5vl:7b"
)

# PDF 파일 처리
result = processor.process_single_pdf("your_document.pdf")

# 결과를 파일로 저장
processor.save_results_to_files(result, "output_directory")
```

### 처리 결과 구조

```python
{
    'source': 'PDF 파일 경로',
    'chunks': [
        # 텍스트 청크 객체들
    ],
    'tables': [
        {
            'index': 0,
            'headers': ['컬럼1', '컬럼2', ...],
            'data': [['값1', '값2', ...], ...],
            'row_count': 5,
            'column_count': 3,
            'raw_content': '원본 마크다운 테이블'
        }
    ],
    'images': [
        {
            'index': 0,
            'caption': '이미지 캡션',
            'description': 'AI 생성 설명',
            'docling_description': 'Docling 원본 설명',
            'ollama_description': 'Ollama AI 설명',
            'image_data': 'base64_encoded_image_data'
        }
    ],
    'stats': {
        'chunk_count': 10,
        'table_count': 2,
        'image_count': 3
    }
}
```

## 고급 설정

### 이미지 추출만 사용 (AI 설명 없이)

```python
processor = PDFProcessor(
    enable_image_extraction=True,
    enable_ollama_description=False
)
```

### 텍스트 청킹만 사용

```python
processor = PDFProcessor(
    enable_image_extraction=False,
    enable_ollama_description=False
)
```

### 다른 Ollama 모델 사용

```python
processor = PDFProcessor(
    enable_image_extraction=True,
    enable_ollama_description=True,
    ollama_model="llava:7b"  # 다른 비전 모델
)
```

## 출력 파일 설명

처리 완료 후 다음과 같은 파일들이 생성됩니다:

- `{filename}_chunks.txt`: 추출된 텍스트 청크들
- `{filename}_tables.json`: 표 데이터 (JSON 형식)
- `{filename}_tables.txt`: 표 데이터 (텍스트 형식)
- `{filename}_images.json`: 이미지 메타데이터
- `{filename}_images.txt`: 이미지 설명 (텍스트 형식)
- `{filename}_stats.json`: 처리 통계 정보

## API 참조

### PDFProcessor 클래스

#### 생성자
```python
PDFProcessor(
    enable_image_extraction: bool = True,
    enable_ollama_description: bool = True,
    ollama_model: str = "qwen2.5vl:7b"
)
```

#### 주요 메서드

- `process_single_pdf(source: str)`: PDF 파일을 처리하여 결과 반환
- `save_results_to_files(results, output_dir)`: 처리 결과를 파일로 저장
- `extract_tables_and_images(doc)`: 문서에서 표와 이미지 추출

### OllamaImageDescriber 클래스

```python
OllamaImageDescriber(model_name: str = "qwen2.5vl:7b")
```

- `describe_image(image_data: bytes, prompt: str = None)`: 이미지 설명 생성

## 문제 해결

### 1. Ollama 연결 오류

```bash
# Ollama 서버 상태 확인
ollama serve

# 다른 포트로 실행
OLLAMA_HOST=0.0.0.0:11435 ollama serve
```

### 2. 모델 다운로드 문제

```bash
# 모델 목록 확인
ollama list

# 모델 재다운로드
ollama pull qwen2.5vl:7b
```

### 3. 메모리 부족

대용량 PDF 파일 처리 시 메모리 부족이 발생할 수 있습니다:

```python
# 이미지 추출 비활성화로 메모리 절약
processor = PDFProcessor(enable_image_extraction=False)
```

### 4. 이미지 추출 실패

일부 PDF에서 이미지 추출이 실패할 수 있습니다. 이는 PDF 인코딩 방식에 따른 것으로 정상적인 동작입니다.

## 성능 최적화

### 1. 배치 처리

```python
def process_multiple_pdfs(pdf_files, output_base_dir):
    processor = PDFProcessor()
    results = []
    
    for pdf_file in pdf_files:
        result = processor.process_single_pdf(pdf_file)
        results.append(result)
        
        # 각 파일별로 별도 디렉토리에 저장
        filename = os.path.basename(pdf_file).replace('.pdf', '')
        output_dir = os.path.join(output_base_dir, filename)
        processor.save_results_to_files(result, output_dir)
    
    return results
```

### 2. 병렬 처리 (고급)

```python
from concurrent.futures import ThreadPoolExecutor
import threading

class ParallelPDFProcessor:
    def __init__(self, max_workers=2):
        self.max_workers = max_workers
        self._lock = threading.Lock()
    
    def process_files(self, pdf_files):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            processor = PDFProcessor()
            futures = [
                executor.submit(processor.process_single_pdf, pdf_file)
                for pdf_file in pdf_files
            ]
            
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=300)  # 5분 타임아웃
                    results.append(result)
                except Exception as e:
                    print(f"처리 실패: {e}")
                    results.append(None)
            
            return results
```
