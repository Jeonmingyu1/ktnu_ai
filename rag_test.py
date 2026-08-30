import os
import chromadb
import pandas as pd
import streamlit as st
from google import genai

# 1. 페이지 설정
st.set_page_config(
    page_title="건축기사 RAG 학습 및 채점 시스템",
    layout="wide",
)

# 2. API 키 설정 (채점 및 답변 생성용)
try:
  GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
  st.error("Streamlit Secrets에 'GEMINI_API_KEY'가 설정되지 않았습니다.")
  st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)


# 3. ChromaDB 초기화 (내장 임베딩 활용으로 API 에러 원천 차단)
@st.cache_resource
def init_chroma():
  chroma_client = chromadb.PersistentClient(path="./chroma_db_final")
  # ChromaDB 기본 임베딩 모델 사용 (추가 API 호출 없이 안정적으로 동작)
  collection = chroma_client.get_or_create_collection(
      name="architectural_exam_semantic"
  )
  return chroma_client, collection


chroma_client, collection = init_chroma()


# 4. 데이터 적재 함수
@st.cache_resource
def load_and_vectorize_data():
  if collection.count() == 0:
    try:
      df = pd.read_csv("architectural_exam.csv")
      df.columns = df.columns.str.strip()
    except FileNotFoundError:
      st.error("architectural_exam.csv 파일이 없습니다!")
      st.stop()

    documents = (
        df["question"].fillna("")
        + " "
        + df["answer"].fillna("")
        + " "
        + df.get("category", "").fillna("")
    ).tolist()
    ids = [str(i) for i in range(len(df))]
    metadatas = df.to_dict(orient="records")

    # ChromaDB가 자동으로 텍스트를 벡터화하여 저장
    collection.add(documents=documents, ids=ids, metadatas=metadatas)


load_and_vectorize_data()

# 5. UI 화면 구성
st.title("🏗️ 건축기사 실기 RAG 학습 및 자동 채점 시스템")
st.write(
    "ChromaDB 벡터 검색과 제미나이 AI 채점을 결합한 의미 기반 시스템입니다."
)

menu = st.sidebar.selectbox(
    "선택 메뉴", ["문제 풀기 & AI 채점", "RAG 검색 테스트"]
)

if menu == "문제 풀기 & AI 채점":
  st.subheader("📝 주관식 서술형 문제 풀이 및 채점")

  user_question = st.text_input(
      "풀고 싶은 문제 키워드나 질문을 입력하세요:", "BOT 방식"
  )
  user_answer = st.text_area(
      "작성한 답안을 입력하세요:",
      "민간이 시설을 짓고 운영한 뒤 국가에 양도하는 방식입니다.",
  )

  if st.button("AI 채점 요청하기"):
    with st.spinner("RAG로 교재를 검색하고 정밀 채점 중입니다..."):
      # ChromaDB 벡터 검색 실행 (의미 기반 유사 문서 탐색)
      search_results = collection.query(query_texts=[user_question], n_results=1)

      retrieved_context = (
          search_results["documents"][0][0]
          if search_results["documents"]
          else "관련 정보 없음"
      )
      metadata = (
          search_results["metadatas"][0][0]
          if search_results["metadatas"]
          else {}
      )

      correct_answer = metadata.get("answer", "정보 없음")

      # AI 채점 프롬프트 실행
      prompt = f"""
            당신은 엄격하고 공정한 건축기사 실기 시험 채점위원입니다.
            아래의 [모범 정답 및 교재 내용]을 바탕으로 [학생의 답안]을 평가하고 점수를 매기세요.

            [모범 정답 및 참고 내용]:
            {correct_answer}

            [학생의 질문/문제]: {user_question}
            [학생의 답안]: {user_answer}

            [채점 루브릭 기준]:
            - 필수 키워드가 들어가고 설명이 정확하면: 90~100점
            - 핵심 키워드는 들어갔으나 설명이 미흡하면: 60~80점
            - 유사한 개념이면: 30~50점
            - 완전히 틀렸거나 공란이면: 0점

            반드시 아래 형식으로 출력할 것:
            1. 점수: (0~100점 사이 숫자)
            2. 채점 피드백 및 감점 요인:
            3. 모범 답안:
            """

      response = client.models.generate_content(
          model="gemini-3.6-flash",
          contents=prompt,
      )

      st.markdown("### 📊 채점 결과")
      st.write(response.text)

      with st.expander("🔍 RAG 검색에 활용된 교재 원문 확인"):
        st.write(retrieved_context)

elif menu == "RAG 검색 테스트":
  st.subheader("🔎 벡터 검색 검증")
  query_text = st.text_input(
      "검색할 내용을 입력하세요 (오타나 평소 말투 가능):",
      "민간이 시설 짓고 국가에 넘기는 계약",
  )
  if st.button("문서 검색"):
    results = collection.query(query_texts=[query_text], n_results=2)
    st.write("검색된 관련 문서:", results["documents"])
