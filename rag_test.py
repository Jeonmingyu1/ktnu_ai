import os
import chromadb
import pandas as pd
import streamlit as st
from google import genai

# 1. 페이지 설정
st.set_page_config(
    page_title="건축기사 RAG 학습 및 채점 시스템 (최종본)",
    layout="wide",
)

# 2. API 키 설정
try:
  GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
  st.error("Streamlit Secrets에 'GEMINI_API_KEY'가 설정되지 않았습니다.")
  st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# 3. ChromaDB 초기화
@st.cache_resource
def init_chroma():
  chroma_client = chromadb.PersistentClient(path="./chroma_db_final")
  return chroma_client


chroma_client = init_chroma()
collection = chroma_client.get_or_create_collection(
    name="architectural_exam_semantic"
)


# 4. 딥러닝 임베딩(text-embedding-004)을 이용한 데이터 적재
@st.cache_resource
def load_and_vectorize_data():
  if collection.count() == 0:
    try:
      df = pd.read_csv("architectural_exam.csv")
    except FileNotFoundError:
      df = pd.DataFrame({
          "question": [
              "피복두께의 목적에 대해 쓰시오",
              "BOT 방식을 설명하시오.",
          ],
          "answer": [
              "내화 확보 및 철근 부식 방지",
              "민간이 시설을 짓고 운영한 뒤 국가에 양도",
          ],
          "explanation": [
              (
                  "피복두께는 콘크리트 중성화 방지 및 내화성 유지를 위해"
                  " 필요합니다."
              ),
              "Social Infrastructure 사업에 주로 쓰입니다.",
          ],
      })

    documents = (
        df["question"] + " " + df["answer"] + " " + df["explanation"]
    ).tolist()
    ids = [str(i) for i in range(len(df))]
    metadatas = df.to_dict(orient="records")

    # 구글 공식 딥러닝 임베딩 모델(text-embedding-004)로 벡터 추출
    response = client.models.embed_content(
        model="text-embedding-004", contents=documents
    )
    embeddings = [e.values for e in response.embeddings]

    # ChromaDB에 임베딩 벡터 저장
    collection.add(
        embeddings=embeddings, documents=documents, ids=ids, metadatas=metadatas
    )


load_and_vectorize_data()

# 5. UI 화면 구성
st.title("🏗️ 건축기사 실기 딥러닝 RAG 학습 및 자동 채점 시스템")
st.write(
    "구글 `text-embedding-004` 딥러닝 모델과 ChromaDB 벡터 검색을 결합한"
    " 완전한 의미 기반(Semantic Search) 시스템입니다."
)

menu = st.sidebar.selectbox(
    "선택 메뉴", ["문제 풀기 & AI 채점", "의미 기반 RAG 검색 테스트"]
)

if menu == "문제 풀기 & AI 채점":
  st.subheader("📝 주관식 서술형 문제 풀이 및 채점")

  user_question = st.text_input(
      "풀고 싶은 문제 키워드나 질문을 입력하세요:", "피복두께의 목적"
  )
  user_answer = st.text_area(
      "작성한 답안을 입력하세요:",
      "철근 부식 방지와 내화성 확보를 위해서입니다.",
  )

  if st.button("AI 채점 요청하기"):
    with st.spinner("딥러닝 임베딩으로 교재를 의미 기반 검색 중..."):
      # 사용자 질문을 딥러닝 벡터로 변환
      query_embedding = client.models.embed_content(
          model="text-embedding-004", contents=user_question
      ).embeddings[0].values

      # 벡터 데이터베이스에서 가장 유사한 문서 검색
      search_results = collection.query(
          query_embeddings=[query_embedding], n_results=1
      )

      retrieved_context = (
          search_results["documents"][0][0]
          if search_results["documents"]
          else "관련 정보 없음"
      )

      # AI 채점 프롬프트 실행
      prompt = f"""
            당신은 엄격하고 공정한 건축기사 실기 시험 채점위원입니다.
            아래의 [참고 교재 내용]을 바탕으로 [학생의 답안]을 평가하고 점수를 매기세요.

            [참고 교재 내용]:
            {retrieved_context}

            [학생의 질문]: {user_question}
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

elif menu == "의미 기반 RAG 검색 테스트":
  st.subheader("🔎 의미 기반(Semantic) 벡터 검색 검증")
  query_text = st.text_input(
      "검색할 내용을 입력하세요 (오타나 평소 말투 가능):", "철근 부식 막는 이유"
  )
  if st.button("문서 검색"):
    q_embed = client.models.embed_content(
        model="text-embedding-004", contents=query_text
    ).embeddings[0].values

    results = collection.query(query_embeddings=[q_embed], n_results=2)
    st.write("검색된 관련 문서:", results["documents"])
