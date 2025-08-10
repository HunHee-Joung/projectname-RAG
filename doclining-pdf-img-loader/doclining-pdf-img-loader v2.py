"""
Docling을 사용한 PDF 파일 청킹, 표 및 이미지 추출 샘플 코드 (Ollama qwen2.5vl:7b 사용)
필요한 패키지 설치: 
- pip install docling
- pip install ollama
- pip install pillow  # 이미지 처리용
- ollama pull qwen2.5vl:7b (터미널에서 실행)
"""

import os
import json
import base64
import io
from pathlib import Path
from typing import List, Optional, Dict, Any

import ollama
from PIL import Image
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HierarchicalChunker


class OllamaImageDescriber:
    """Ollama qwen2.5vl:7b을 사용한 이미지 설명 생성기"""
    
    def __init__(self, model_name: str = "qwen2.5vl:7b"):
        self.model_name = model_name
        self.client = ollama.Client()
        self._check_model_availability()
    
    def _check_model_availability(self):
        """모델이 사용 가능한지 확인"""
        try:
            models = self.client.list()
            available_models = []
            
            if 'models' in models:
                for model in models['models']:
                    if isinstance(model, dict):
                        if 'name' in model:
                            available_models.append(model['name'])
                        elif 'model' in model:
                            available_models.append(model['model'])
                    else:
                        available_models.append(str(model))
            
            # 모델 이름 매칭
            model_found = False
            for available_model in available_models:
                if (self.model_name in available_model or 
                    available_model.startswith(self.model_name) or
                    available_model == f"{self.model_name}:latest"):
                    print(f"✅ 모델 발견: {available_model}")
                    model_found = True
                    break
            
            if not model_found:
                print(f"경고: {self.model_name} 모델이 설치되지 않았습니다.")
                print(f"다음 명령어로 모델을 설치하세요: ollama pull {self.model_name}")
                raise Exception(f"Model {self.model_name} not found")
            
            print(f"Ollama {self.model_name} 모델이 준비되었습니다.")
            
        except Exception as e:
            print(f"Ollama 연결 오류: {e}")
            print("Ollama 서버가 실행 중인지 확인하세요: ollama serve")
            raise
    
    def describe_image(self, image_data: bytes, prompt: str = None) -> str:
        """이미지 설명 생성"""
        if prompt is None:
            prompt = "이 이미지를 3-5문장으로 정확하고 간결하게 설명해주세요. 한국어로 답변해주세요."
        
        try:
            if not isinstance(image_data, bytes):
                raise ValueError(f"이미지 데이터는 bytes 타입이어야 합니다. 현재 타입: {type(image_data)}")
            
            if len(image_data) == 0:
                raise ValueError("이미지 데이터가 비어있습니다.")
            
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [image_base64]
                    }
                ]
            )
            
            return response['message']['content']
            
        except Exception as e:
            print(f"이미지 설명 생성 오류: {e}")
            return f"이미지 설명 생성 실패: {str(e)}"


class PDFProcessor:
    """PDF 파일을 청킹하고 표/이미지를 추출하는 클래스"""
    
    def __init__(self, enable_image_extraction: bool = True, enable_ollama_description: bool = True, 
                 ollama_model: str = "qwen2.5vl:7b"):
        self.enable_image_extraction = enable_image_extraction
        self.enable_ollama_description = enable_ollama_description
        self.converter = self._setup_converter()
        self.chunker = HierarchicalChunker()
        
        if self.enable_ollama_description:
            try:
                self.image_describer = OllamaImageDescriber(ollama_model)
                print("✅ Ollama 이미지 설명 기능이 활성화되었습니다.")
            except Exception as e:
                print(f"⚠️ Ollama 초기화 실패: {e}")
                print("이미지 설명 기능이 비활성화됩니다. 다른 기능은 정상 작동합니다.")
                self.enable_ollama_description = False
                self.image_describer = None
        else:
            self.image_describer = None
    
    def _setup_converter(self) -> DocumentConverter:
        """DocumentConverter 설정"""
        if self.enable_image_extraction:
            pipeline_options = PdfPipelineOptions(
                do_picture_description=False,
                generate_picture_images=True,
                images_scale=2,
            )
            
            return DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
        else:
            return DocumentConverter()
    
    def _extract_from_pil_image(self, img_obj, img_num):
        """PIL Image 방식으로 바이트 추출"""
        try:
            if hasattr(img_obj, 'pil_image'):
                pil_img = img_obj.pil_image
                if pil_img is not None:
                    img_buffer = io.BytesIO()
                    save_format = 'PNG'
                    if hasattr(pil_img, 'format') and pil_img.format:
                        save_format = pil_img.format
                    pil_img.save(img_buffer, format=save_format)
                    return img_buffer.getvalue()
        except Exception as e:
            print(f"이미지 {img_num}: PIL Image 추출 실패: {e}")
        return None
    
    def _extract_from_data_attr(self, img_obj, img_num):
        """data 속성에서 바이트 추출"""
        try:
            if hasattr(img_obj, 'data'):
                data = img_obj.data
                if isinstance(data, bytes) and len(data) > 0:
                    return data
        except Exception as e:
            print(f"이미지 {img_num}: data 속성 추출 실패: {e}")
        return None
    
    def _extract_from_bytes_attrs(self, img_obj, img_num):
        """다양한 바이트 속성에서 추출"""
        byte_attrs = ['bytes', 'raw_data', 'content', 'buffer', '_data', 'image_data', 'binary_data']
        
        for attr in byte_attrs:
            try:
                if hasattr(img_obj, attr):
                    potential_bytes = getattr(img_obj, attr)
                    if isinstance(potential_bytes, bytes) and len(potential_bytes) > 0:
                        print(f"이미지 {img_num}: {attr} 속성에서 바이트 데이터 발견")
                        return potential_bytes
            except Exception as e:
                print(f"이미지 {img_num}: {attr} 속성 접근 실패: {e}")
                continue
        return None
    
    def _extract_from_methods(self, img_obj, img_num):
        """다양한 메서드로 바이트 추출"""
        method_names = ['to_bytes', 'get_bytes', 'read', 'getvalue', 'tobytes', 'as_bytes']
        
        for method_name in method_names:
            try:
                if hasattr(img_obj, method_name):
                    method = getattr(img_obj, method_name)
                    if callable(method):
                        result = method()
                        if isinstance(result, bytes) and len(result) > 0:
                            print(f"이미지 {img_num}: {method_name}() 메서드로 바이트 데이터 획득")
                            return result
            except Exception as e:
                print(f"이미지 {img_num}: {method_name}() 메서드 호출 실패: {e}")
                continue
        return None
    
    def extract_tables_and_images(self, doc) -> Dict[str, Any]:
        """문서에서 표와 이미지 추출"""
        extracted_data = {
            'tables': [],
            'images': [],
            'image_descriptions': []
        }
        
        try:
            # 마크다운으로 변환하여 표 추출
            markdown_content = doc.export_to_markdown()
            tables = self._extract_tables_from_markdown(markdown_content)
            extracted_data['tables'] = tables
            
            # 이미지 추출
            if hasattr(doc, 'pictures') and doc.pictures:
                print(f"PDF에서 {len(doc.pictures)}개의 이미지를 발견했습니다.")
                
                for i, picture in enumerate(doc.pictures):
                    print(f"\n=== 이미지 {i+1} 처리 시작 ===")
                    
                    image_info = {
                        'index': i,
                        'caption': getattr(picture, 'caption', ''),
                        'description': '',
                        'docling_description': getattr(picture, 'description', ''),
                        'image_data': None,
                        'ollama_description': ''
                    }
                    
                    image_bytes = None
                    
                    if hasattr(picture, 'image') and picture.image:
                        print(f"이미지 {i+1} 객체 존재 확인: True")
                        
                        try:
                            img_obj = picture.image
                            print(f"이미지 {i+1} 타입: {type(img_obj).__name__}")
                            
                            # 속성 목록
                            try:
                                attrs = [attr for attr in dir(img_obj) if not attr.startswith('_')]
                                print(f"이미지 {i+1} 공개 속성들: {attrs[:10]}...")
                            except Exception as attr_error:
                                print(f"이미지 {i+1} 속성 조회 실패: {attr_error}")
                            
                        except Exception as basic_error:
                            print(f"이미지 {i+1} 기본 정보 조회 실패: {basic_error}")
                            img_obj = picture.image
                        
                        # 이미지 데이터 추출 시도
                        extraction_methods = [
                            ('pil_image', lambda obj: self._extract_from_pil_image(obj, i+1)),
                            ('data', lambda obj: self._extract_from_data_attr(obj, i+1)),
                            ('bytes_attr', lambda obj: self._extract_from_bytes_attrs(obj, i+1)),
                            ('methods', lambda obj: self._extract_from_methods(obj, i+1)),
                        ]
                        
                        for method_name, extraction_func in extraction_methods:
                            if image_bytes is not None:
                                break
                                
                            try:
                                print(f"이미지 {i+1}: {method_name} 방식 시도...")
                                result = extraction_func(img_obj)
                                if result and isinstance(result, bytes) and len(result) > 0:
                                    image_bytes = result
                                    print(f"이미지 {i+1}: {method_name} 방식 성공! 크기: {len(image_bytes)} bytes")
                                    break
                                else:
                                    print(f"이미지 {i+1}: {method_name} 방식 실패 또는 빈 데이터")
                            except Exception as extract_error:
                                print(f"이미지 {i+1}: {method_name} 방식 예외: {extract_error}")
                                continue
                        
                        # base64 인코딩
                        if image_bytes and isinstance(image_bytes, bytes) and len(image_bytes) > 0:
                            try:
                                image_info['image_data'] = base64.b64encode(image_bytes).decode('utf-8')
                                print(f"이미지 {i+1}: base64 인코딩 성공")
                            except Exception as b64_error:
                                print(f"이미지 {i+1}: base64 인코딩 실패: {b64_error}")
                                image_bytes = None
                        else:
                            print(f"이미지 {i+1}: 모든 추출 방식 실패")
                    else:
                        print(f"이미지 {i+1}: 이미지 객체가 없음")
                    
                    # Ollama 설명 생성
                    if self.enable_ollama_description and self.image_describer and image_bytes:
                        try:
                            print(f"이미지 {i+1}: Ollama 설명 생성 시작 (크기: {len(image_bytes)} bytes)")
                            
                            ollama_desc = self.image_describer.describe_image(image_bytes)
                            image_info['ollama_description'] = ollama_desc
                            image_info['description'] = ollama_desc
                            print(f"이미지 {i+1}: Ollama 설명 완료 - {ollama_desc[:100]}...")
                            
                        except Exception as e:
                            print(f"이미지 {i+1}: Ollama 설명 생성 실패: {e}")
                            image_info['ollama_description'] = f"설명 생성 실패: {str(e)}"
                    elif image_bytes is None:
                        print(f"이미지 {i+1}: 바이트 데이터가 없어 Ollama 설명 생성 생략")
                    
                    # Docling 원본 설명 사용 (Ollama 설명이 없는 경우)
                    if not image_info['description'] and image_info['docling_description']:
                        image_info['description'] = image_info['docling_description']
                    
                    print(f"=== 이미지 {i+1} 처리 완료 ===\n")
                    extracted_data['images'].append(image_info)
            
            print(f"추출 완료: 표 {len(extracted_data['tables'])}개, 이미지 {len(extracted_data['images'])}개")
            
        except Exception as e:
            print(f"표/이미지 추출 중 오류: {e}")
        
        return extracted_data
    
    def _extract_tables_from_markdown(self, markdown_content: str) -> List[Dict[str, Any]]:
        """마크다운 내용에서 표 추출"""
        tables = []
        lines = markdown_content.split('\n')
        current_table = []
        in_table = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            if line.startswith('|') and line.endswith('|'):
                if not in_table:
                    in_table = True
                    current_table = []
                
                if not all(c in '|-: ' for c in line):
                    current_table.append(line)
            else:
                if in_table and current_table:
                    table_info = self._parse_markdown_table(current_table, len(tables))
                    if table_info:
                        tables.append(table_info)
                    current_table = []
                in_table = False
        
        if current_table:
            table_info = self._parse_markdown_table(current_table, len(tables))
            if table_info:
                tables.append(table_info)
        
        return tables
    
    def _parse_markdown_table(self, table_lines: List[str], table_index: int) -> Optional[Dict[str, Any]]:
        """마크다운 테이블 라인들을 파싱"""
        if not table_lines:
            return None
        
        try:
            rows = []
            for line in table_lines:
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                if cells:
                    rows.append(cells)
            
            if not rows:
                return None
            
            return {
                'index': table_index,
                'headers': rows[0] if rows else [],
                'data': rows[1:] if len(rows) > 1 else [],
                'row_count': len(rows),
                'column_count': len(rows[0]) if rows else 0,
                'raw_content': '\n'.join(table_lines)
            }
            
        except Exception as e:
            print(f"테이블 파싱 오류: {e}")
            return None
    
    def process_single_pdf(self, source: str) -> Dict[str, Any]:
        """단일 PDF 파일을 처리"""
        try:
            print(f"PDF 처리 시작: {source}")
            
            result = self.converter.convert(source=source)
            doc = result.document
            
            print(f"PDF 변환 완료. 문서 변환 성공: {result.status}")
            
            print("문서 청킹 시작...")
            doc_chunks = list(self.chunker.chunk(doc))
            
            print("표 및 이미지 추출 시작...")
            extracted_data = self.extract_tables_and_images(doc)
            
            result_data = {
                'source': source,
                'chunks': doc_chunks,
                'tables': extracted_data['tables'],
                'images': extracted_data['images'],
                'image_descriptions': extracted_data['image_descriptions'],
                'stats': {
                    'chunk_count': len(doc_chunks),
                    'table_count': len(extracted_data['tables']),
                    'image_count': len(extracted_data['images'])
                }
            }
            
            print(f"처리 완료 - 청크: {result_data['stats']['chunk_count']}개, "
                  f"표: {result_data['stats']['table_count']}개, "
                  f"이미지: {result_data['stats']['image_count']}개")
            
            return result_data
            
        except Exception as e:
            print(f"PDF 처리 중 오류 발생: {str(e)}")
            return {
                'source': source,
                'chunks': [],
                'tables': [],
                'images': [],
                'image_descriptions': [],
                'stats': {'chunk_count': 0, 'table_count': 0, 'image_count': 0},
                'error': str(e)
            }
    
    def save_results_to_files(self, results: Dict[str, Any], output_dir: str = "output"):
        """처리 결과를 파일들로 저장"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            source_name = os.path.basename(results['source']).replace('.pdf', '')
            
            # 청크 저장
            if results['chunks']:
                chunks_file = os.path.join(output_dir, f"{source_name}_chunks.txt")
                with open(chunks_file, 'w', encoding='utf-8') as f:
                    for i, chunk in enumerate(results['chunks']):
                        f.write(f"=== 청크 {i+1} ===\n")
                        f.write(f"텍스트: {chunk.text}\n")
                        if hasattr(chunk, 'meta') and chunk.meta:
                            f.write(f"메타데이터: {chunk.meta}\n")
                        f.write("\n" + "="*50 + "\n\n")
                print(f"청크가 {chunks_file}에 저장되었습니다.")
            
            # 표 저장
            if results['tables']:
                tables_file = os.path.join(output_dir, f"{source_name}_tables.json")
                with open(tables_file, 'w', encoding='utf-8') as f:
                    json.dump(results['tables'], f, ensure_ascii=False, indent=2)
                print(f"표 데이터가 {tables_file}에 저장되었습니다.")
                
                tables_txt_file = os.path.join(output_dir, f"{source_name}_tables.txt")
                with open(tables_txt_file, 'w', encoding='utf-8') as f:
                    for table in results['tables']:
                        f.write(f"=== 표 {table['index'] + 1} ===\n")
                        f.write(f"행 수: {table['row_count']}, 열 수: {table['column_count']}\n")
                        f.write(f"헤더: {', '.join(table['headers'])}\n")
                        f.write("데이터:\n")
                        for row in table['data']:
                            f.write(f"  {' | '.join(row)}\n")
                        f.write("\n원본 내용:\n")
                        f.write(table['raw_content'])
                        f.write("\n" + "="*50 + "\n\n")
                print(f"표 텍스트가 {tables_txt_file}에 저장되었습니다.")
            
            # 이미지 정보 저장
            if results['images']:
                images_file = os.path.join(output_dir, f"{source_name}_images.json")
                images_meta = []
                for img in results['images']:
                    img_meta = {k: v for k, v in img.items() if k != 'image_data'}
                    img_meta['has_image_data'] = bool(img.get('image_data'))
                    images_meta.append(img_meta)
                
                with open(images_file, 'w', encoding='utf-8') as f:
                    json.dump(images_meta, f, ensure_ascii=False, indent=2)
                print(f"이미지 정보가 {images_file}에 저장되었습니다.")
                
                images_txt_file = os.path.join(output_dir, f"{source_name}_images.txt")
                with open(images_txt_file, 'w', encoding='utf-8') as f:
                    for img in results['images']:
                        f.write(f"=== 이미지 {img['index'] + 1} ===\n")
                        f.write(f"캡션: {img['caption'] or '없음'}\n")
                        f.write(f"Docling 설명: {img['docling_description'] or '없음'}\n")
                        f.write(f"Ollama 설명: {img['ollama_description'] or '없음'}\n")
                        f.write(f"최종 설명: {img['description'] or '없음'}\n")
                        f.write(f"이미지 데이터: {'있음' if img['image_data'] else '없음'}\n")
                        f.write("\n" + "="*50 + "\n\n")
                print(f"이미지 텍스트가 {images_txt_file}에 저장되었습니다.")
            
            # 통계 정보 저장
            stats_file = os.path.join(output_dir, f"{source_name}_stats.json")
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(results['stats'], f, ensure_ascii=False, indent=2)
            print(f"통계 정보가 {stats_file}에 저장되었습니다.")
            
        except Exception as e:
            print(f"파일 저장 중 오류 발생: {str(e)}")


def test_ollama_connection():
    """Ollama 연결 및 모델 테스트"""
    print("=== Ollama 연결 테스트 ===")
    
    try:
        client = ollama.Client()
        models = client.list()
        print(f"✅ Ollama 서버 연결 성공")
        
        try:
            describer = OllamaImageDescriber("qwen2.5vl:7b")
            print("✅ qwen2.5vl:7b 모델 사용 가능")
        except Exception as model_error:
            print(f"⚠️ qwen2.5vl:7b 모델 문제: {model_error}")
            print("다음 명령어로 모델을 설치하세요: ollama pull qwen2.5vl:7b")
        
        print("\nOllama 설정이 완료되었습니다.")
        
    except Exception as e:
        print(f"❌ Ollama 연결 실패: {e}")
        print("해결 방법:")
        print("1. Ollama 서버 실행: ollama serve")
        print("2. 모델 다운로드: ollama pull qwen2.5vl:7b")


def main():
    """메인 함수"""
    processor = PDFProcessor(
        enable_image_extraction=True, 
        enable_ollama_description=True,
        ollama_model="qwen2.5vl:7b"
    )
    
    print("=== 단일 PDF 처리 예시 ===")
    # pdf_url = "https://arxiv.org/pdf/2311.18481"
    pdf_url = "./SPRi AI Brief_3월호_산업동향_F.pdf"
    result = processor.process_single_pdf(pdf_url)
    
    if result['chunks']:
        print(f"\n첫 번째 청크 내용 (처음 500자):")
        print(result['chunks'][0].text[:500] + "...")
        
        if result['tables']:
            print(f"\n추출된 표 정보:")
            for table in result['tables'][:3]:
                print(f"  표 {table['index'] + 1}: {table['row_count']}행 x {table['column_count']}열")
        
        if result['images']:
            print(f"\n추출된 이미지 정보:")
            for img in result['images'][:3]:
                print(f"  이미지 {img['index'] + 1}:")
                print(f"    Ollama 설명: {img['ollama_description'][:150]}..." if img['ollama_description'] else "    Ollama 설명: 없음")
                print(f"    이미지 데이터: {'있음' if img['image_data'] else '없음'}")
        
        processor.save_results_to_files(result, "pdf_output")


if __name__ == "__main__":
    test_ollama_connection()
    print("\n" + "="*60 + "\n")
    main()
