import os
import chromadb
import pandas as pd
import streamlit as st
from google import genai

# set_page_config 중복/순서 에러 방지용 안전 장치
try:
  st.set_page_config(
      page_title="건축기사 RAG 학습 및 채점 시스템", page_layout="wide"
  )
except Exception:
  pass

# Secrets에서 API 키 로드
try:
  GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
  st.error(
      "Streamlit Secrets에 'GEMINI_API_KEY'가 설정되지 않았습니다. 설정 후"
      " 다시 시도해주세요."
  )
  st.stop()

# 제미나이 클라이언트 초기화
client = genai.Client(api_key=GEMINI_API_KEY)


# ChromaDB 영구 저장소 설정
@st.cache_resource
def init_chroma():
  return chromadb.PersistentClient(path="./chroma_db")


chroma_client = init_chroma()
collection = chroma_client.get_or_create_collection(
    name="architectural_exam_collection"
)


# 데이터 적재 함수
def load_and_vectorize_data():
  if collection.count() == 0:
    try:
      df = pd.read_csv("architectural_exam.csv")
      documents = df["question"] + " " + df["answer"] + " " + df["explanation"]
      ids = [str(i) for i in range(len(df))]
      metadatas = df.to_dict(orient="records")
      collection.add(
          documents=documents.tolist(), ids=ids, metadatas=metadatas
      )
    except Exception:
      sample_docs = [
          "철근콘크리트 구조의 피복두께는 철근의 부식을 방지하고 내화를 확보하기 위함이다.",
          "네트워크 공정표에서 주공정선(Critical Path)은 전체 공사 기간을 결정하는 최장 경로이다.",
      ]
      collection.add(
          documents=sample_docs,
          ids=["1", "2"],
          metadatas=[
              {"question": "피복두께의 목적", "answer": "내화 및 부식 방지"},
              {
                  "question": "주공정선의 정의",
                  "answer": "전체 공사 기간을 결정하는 최장 경로",
              },
          ],
      )


load_and_vectorize_data()

# 메인 UI
st.title("🏗️ 건축기사 실기 RAG 학습 및 자동 채점 시스템")
st.write(
    "딥러닝 임베딩과 RAG 기술을 활용하여 기출문제를 검색하고, 객관적인 루브릭에"
    " 따라 답안을 채점합니다."
)

menu = st.sidebar.selectbox("선택 메뉴", ["문제 풀기 & AI 채점", "RAG 검색 테스트"])

if menu == "문제 풀기 & AI 채점":
  st.subheader("📝 주관식 서술형 문제 풀이 및 채점")
  user_question = st.text_input(
      "풀고 싶은 문제 키워드나 질문을 입력하세요:", "피복두께의 목적에 대해 쓰시오"
  )
  user_answer = st.text_area(
      "작성한 답안을 입력하세요:",
      "철근의 부식을 방지하고 내화성을 확보하기 위해서입니다.",
  )

  if st.button("AI 채점 요청하기"):
    with st.spinner("RAG로 교재를 검색하고 정밀 채점 중입니다..."):
      search_results = collection.query(
          query_texts=[user_question], n_results=1
      )
      retrieved_context = (
          search_results["documents"][0][0]
          if search_results["documents"]
          else "관련 정보 없음"
      )

      prompt = f"""
            당신은 엄격하고 공정한 건축기사 실기 시험 채점위원입니다.
            아래의 [참고 교재 내용]을 바탕으로 [학생의 답안]을 평가하고 점수를 매기세요.

            [참고 교재 내용]:
            {retrieved_context}

            [학생의 질문]: {user_question}
            [학생의 답안]: {user_answer}

            [채점 루브릭 기준]:
            - 필수 키워드가 100% 들어가고 설명이 정확하면: 90~100점
            - 핵심 키워드는 들어갔으나 설명이 미흡하면: 60~80점
            - 핵심 키워드가 누락되었으나 유사한 개념이면: 30~50점
            - 완전히 틀렸거나 공란이면: 0점

            반드시 아래 형식으로 출력할 것:
            1. 점수: (0~100점 사이 숫자만)
            2. 채점 피드백 및 감점 요인 상세 설명:
            3. 모범 답안:
            """

      response = client.models.generate_content(
          model="gemini-3.6-flash", contents=prompt
      )

      st.markdown("### 📊 채점 결과")
      st.write(response.text)

      with st.expander("🔍 RAG 검색에 활용된 교재 원문 확인"):
        st.write(retrieved_context)

elif menu == "RAG 검색 테스트":
  st.subheader("🔎 RAG 벡터 검색 검증")
  query_text = st.text_input("검색할 내용을 입력하세요:", "피복두께")
  if st.button("문서 검색"):
    results = collection.query(query_texts=[query_text], n_results=2)
    st.write("검색된 관련 문서:", results["documents"])
