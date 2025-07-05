import os
import logging
import pandas as pd
from qdrant_client import QdrantClient

# ✅ 설정 파일에서 Qdrant 접속 정보 가져오기
# from config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME
QDRANT_HOST = os.getenv("QDRANT_HOST", "192.168.0.249")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docling_search")
MODEL_NAME = "BAAI/bge-m3"

# --- 전역 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QdrantDataViewer:
    """
    Qdrant 컬렉션의 모든 데이터 요약을 표로 출력하는 도구 클래스
    """
    def __init__(self):
        """Qdrant 클라이언트를 초기화합니다."""
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)
            self.client.get_collections()
            logging.info(f"✅ Qdrant 서버에 연결 성공: {QDRANT_HOST}:{QDRANT_PORT}")
        except Exception as e:
            logging.error(f"❌ Qdrant 서버 연결 실패: {e}")
            logging.error("Qdrant 서버가 실행 중인지, 호스트와 포트 정보가 올바른지 확인해주세요.")
            exit(1)

    def display_all_chunks_summary(self):
        """컬렉션의 모든 청크에 대한 요약 정보를 표로 출력합니다."""
        try:
            collection_info = self.client.get_collection(collection_name=COLLECTION_NAME)
            total_points = collection_info.points_count
            logging.info(f"✅ 컬렉션 '{COLLECTION_NAME}' 확인. 총 {total_points}개의 청크가 존재합니다.")

            if total_points == 0:
                logging.warning("컬렉션에 데이터가 없습니다. 먼저 인덱싱을 실행해주세요.")
                return

            logging.info("모든 청크의 요약 정보를 가져오는 중...")
            all_points, _ = self.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=total_points,
                with_payload=True,
                with_vectors=False
            )
            
            summary_data = []
            for point in all_points:
                payload = point.payload
                text_preview = (payload.get("text", "")[:70] + "...") if payload.get("text") else ""
                
                summary_data.append({
                    "ID": point.id,
                    "Type": payload.get("element_type"),
                    "Page": payload.get("page_no"),
                    "Is_Ctx": payload.get("is_contextualized"),
                    "Parent_ID": payload.get("parent_ref"),
                    "Self_ID": payload.get("self_ref"),
                    "Text_Preview": text_preview.replace('\n', ' ')
                })
            
            df = pd.DataFrame(summary_data)
            pd.set_option('display.max_rows', 200)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 250)
            pd.set_option('display.max_colwidth', 75)

            print("\n" + "="*120)
            print(f"📊 컬렉션 '{COLLECTION_NAME}' 전체 청크 요약 정보")
            print("="*120)
            print(df)
            print("="*120)

        except Exception as e:
            logging.error(f"❌ 전체 청크 요약 정보 표시 중 오류 발생: {e}")

    def run(self):
        """데이터 뷰어의 모든 기능을 순차적으로 실행합니다."""
        self.display_all_chunks_summary()
        logging.info("👋 데이터 출력이 완료되었습니다. 프로그램을 종료합니다.")


if __name__ == "__main__":
    viewer = QdrantDataViewer()
    viewer.run()
