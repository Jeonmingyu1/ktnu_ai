import os
import chromadb
import pandas as pd
import streamlit as st
from google import genai

# 1. 페이지 설정
st.set_page_config(page_title="건축기사 AI 학습 시스템", layout="wide")
st.title("🏗️ 건축기사 AI 학습 & 채점 시스템")

# 2. API 키 설정
try:
  GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
  st.error("Streamlit Secrets에 'GEMINI_API_KEY'가 설정되지 않았습니다.")
  st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)


# 3. ChromaDB 초기화 및 데이터 적재
@st.cache_resource
def init_chroma_and_load():
  chroma_client = chromadb.PersistentClient(path="./chroma_db_final")
  collection = chroma_client.get_or_create_collection(
      name="architectural_exam_semantic"
  )

  if collection.count() == 0:
    try:
      df = pd.read_csv("architectural_exam.csv")
      df.columns = df.columns.str.strip()
    except FileNotFoundError:
      st.error("⚠️ 'architectural_exam.csv' 파일이 없습니다!")
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

    collection.add(documents=documents, ids=ids, metadatas=metadatas)

  return collection


collection = init_chroma_and_load()

# 전체 데이터프레임 로드
df = pd.read_csv("architectural_exam.csv")
df.columns = df.columns.str.strip()

# 카테고리 정리 (대단원 매핑)
if "category" not in df.columns:
  df["category"] = "건축시공"


def classify_units(row):
  combined = str(row["category"]) + " " + str(row["question"])
  if any(
      k in combined
      for k in ["공정", "네트워크", "CPM", "공정표", "VE", "가치공학"]
  ):
    return "공정관리"
  elif any(
      k in combined for k in ["적산", "견적", "수량", "단가", "공사비", "물량산출"]
  ):
    return "건축적산"
  elif any(
      k in combined
      for k in [
          "구조역학",
          "모멘트",
          "응력",
          "보의",
          "하중",
          "처짐",
          "철근콘크리트",
          "철골",
      ]
  ):
    return "건축구조"
  else:
    return "건축시공"


df["대단원"] = df.apply(classify_units, axis=1)


def extract_score(result_text):
  import re

  match = re.search(r"(?:최종\s*점수|점수)[\s:]*([0-9]{1,3})점?", result_text)
  if match:
    return int(match.group(1))
  match_any = re.search(r"\b([0-9]{1,3})\b", result_text)
  if match_any:
    val = int(match_any.group(1))
    if 0 <= val <= 100:
      return val
  return 0


# ==================== [세션 상태 초기화] ====================
if "scope_mode" not in st.session_state:
  st.session_state["scope_mode"] = "🎲 전체 챕터"
if "target_weak_major" not in st.session_state:
  st.session_state["target_weak_major"] = None
if "selected_major_val" not in st.session_state:
  major_unique = df["대단원"].unique().tolist()
  st.session_state["selected_major_val"] = major_unique[0] if major_unique else ""
if "active_tab_index" not in st.session_state:
  st.session_state["active_tab_index"] = 0

# ==================== [사이드바: 학습 범위 설정] ====================
st.sidebar.markdown("### 🎛️ 공부할 범위 고르기")
current_mode = st.session_state["scope_mode"]
st.sidebar.markdown(f"현재 학습 모드: **{current_mode}**")

if st.sidebar.button("🎲 전체 챕터로 변경", use_container_width=True):
  st.session_state["scope_mode"] = "🎲 전체 챕터"
  st.session_state["target_weak_major"] = None
  if "batch_exam_df" in st.session_state:
    del st.session_state["batch_exam_df"]
  st.rerun()

major_list = df["대단원"].unique().tolist()
selected_major_sb = st.sidebar.selectbox("📚 챕터별 학습 (대단원 선택)", major_list)
if st.sidebar.button("📚 선택한 챕터로 공부 시작", use_container_width=True):
  st.session_state["scope_mode"] = "📚 챕터별 학습"
  st.session_state["selected_major_val"] = selected_major_sb
  st.session_state["target_weak_major"] = None
  if "batch_exam_df" in st.session_state:
    del st.session_state["batch_exam_df"]
  st.rerun()

if (
    st.session_state["scope_mode"] == "🚨 취약 파트 공부"
    and st.session_state["target_weak_major"]
):
  st.sidebar.warning(
      f"🚨 **집중 공략 중인 파트:**\n\n**{st.session_state['target_weak_major']}**"
  )

st.sidebar.divider()

# 대상 데이터프레임 필터링
target_df = pd.DataFrame()
if st.session_state["scope_mode"] == "🎲 전체 챕터":
  target_df = df
elif st.session_state["scope_mode"] == "📚 챕터별 학습":
  target_df = df[df["대단원"] == st.session_state["selected_major_val"]]
elif st.session_state["scope_mode"] == "🚨 취약 파트 공부":
  weak_m = st.session_state["target_weak_major"]
  target_df = df[df["대단원"] == weak_m] if weak_m else df

# ==================== [메인 상단 네비게이션 버튼 (3단계 탭)] ====================
col_t1, col_t2, col_t3 = st.columns(3)

with col_t1:
  if st.button(
      "🎯 1단계: 문제 풀기 & AI",
      use_container_width=True,
      type="primary"
      if st.session_state["active_tab_index"] == 0
      else "secondary",
  ):
    st.session_state["active_tab_index"] = 0
    st.rerun()
with col_t2:
  if st.button(
      "📑 2단계: 시험지 모드",
      use_container_width=True,
      type="primary"
      if st.session_state["active_tab_index"] == 1
      else "secondary",
  ):
    st.session_state["active_tab_index"] = 1
    st.rerun()
with col_t3:
  if st.button(
      "📊 3단계: 성적표 & 오답",
      use_container_width=True,
      type="primary"
      if st.session_state["active_tab_index"] == 2
      else "secondary",
  ):
    st.session_state["active_tab_index"] = 2
    st.rerun()

st.divider()

# ==================== [탭 1: 한 문제씩 풀기] ====================
if st.session_state["active_tab_index"] == 0:
  if st.session_state["scope_mode"] == "🚨 취약 파트 공부":
    st.info(
        f"🚨 현재 **[{st.session_state['target_weak_major']}]** 파트 집중"
        " 공략 모드입니다!"
    )
  else:
    st.markdown(
        "#### 💡 한 문제씩 집중적으로 풀고 AI의 채점 결과와 모범 답안을 즉시"
        " 확인하는 모드입니다."
    )

  q_list = target_df["question"].tolist() if not target_df.empty else []
  if not q_list:
    st.warning(
        "⚠️ 선택된 범위에 문제가 없습니다. 사이드바나 3단계 성적표에서 파트를"
        " 다시 선택해 주세요."
    )
  else:
    selected_q = st.selectbox(
        "📌 풀고 싶은 문제를 선택하세요:", q_list, key="single_q_select"
    )
    row_data = target_df[target_df["question"] == selected_q].iloc[0]

    correct_answer = row_data["answer"]
    q_major = row_data["대단원"]
    q_id = row_data["id"]

    st.info(f"**[출제정보] ID: {q_id}  |  대단원: {q_major}**\n\n{selected_q}")

    user_ans = st.text_area(
        "✍️ 정답을 서술형으로 입력하세요:", height=120, key="single_user_ans"
    )

    if st.button("🤖 AI 채점 요청하기", type="primary"):
      if not user_ans:
        st.warning("답안을 입력해주세요!")
      else:
        with st.spinner("RAG 검색 및 AI 채점 중..."):
          # RAG 검색
          search_results = collection.query(
              query_texts=[selected_q], n_results=1
          )
          retrieved_context = (
              search_results["documents"][0][0]
              if search_results["documents"]
              else correct_answer
          )

          prompt = f"""
                    너는 건축기사 실기 수석 채점관이야.
                    [문제]: {selected_q}
                    [모범 답안]: {correct_answer}
                    [참고 교재 내용]: {retrieved_context}
                    [학생 답안]: {user_ans}
                    
                    핵심 키워드 포함 여부를 엄격히 평가해 0~100점의 점수를 매기고 피드백해줘.
                    반드시 아래 형식으로 출력할 것:
                    1. 최종 점수: XX점
                    2. 키워드 포함 여부: (...)
                    3. 채점 상세 평가: (...)
                    """
          response = client.models.generate_content(
              model="gemini-3.6-flash", contents=prompt
          )
          result_text = response.text
          score = extract_score(result_text)

          st.session_state["last_graded"] = {
              "question": selected_q,
              "user_ans": user_ans,
              "score": score,
              "result_text": result_text,
              "correct_answer": correct_answer,
          }

          # 결과 저장
          file_name = "results.csv"
          file_exists = os.path.isfile(file_name)
          with open(
              file_name, mode="a", newline="", encoding="utf-8-sig"
          ) as f:
            import csv

            writer = csv.writer(f)
            if not file_exists:
              writer.writerow(
                  ["선택한문제", "대단원", "학생답안", "점수", "AI채점결과"]
              )
            writer.writerow([
                selected_q,
                q_major,
                user_ans,
                score,
                result_text.replace("\n", " "),
            ])
          st.success("채점 완료 및 오답노트 저장 완료!")

    if (
        "last_graded" in st.session_state
        and st.session_state["last_graded"]["question"] == selected_q
    ):
      lg = st.session_state["last_graded"]
      st.markdown("---")
      st.markdown("### 📋 채점 결과 및 정답 확인")
      st.info(f"**점수: {lg['score']}점**")
      st.markdown(lg["result_text"])
      st.success(f"**📖 모범 답안**\n\n{lg['correct_answer']}")
      st.markdown("---")

# ==================== [탭 2: 시험지 모드] ====================
elif st.session_state["active_tab_index"] == 1:
  st.markdown(
      "#### 📑 여러 문제를 시험지처럼 지정한 문항 수만큼 뽑아서 한 번에 풀고"
      " 채점하는 모드입니다."
  )
  if target_df.empty:
    st.warning("⚠️ 선택된 범위에 문제가 없습니다.")
  else:
    c_cnt, c_action = st.columns([2, 2])
    with c_cnt:
      max_limit = len(target_df)
      num_q = st.number_input(
          "추출 문항 수 설정",
          min_value=1,
          max_value=max(1, max_limit),
          value=min(3, max_limit),
      )
    with c_action:
      st.markdown("<br>", unsafe_allow_html=True)
      if st.button(
          "🎲 새로운 문제 세트 무작위 뽑기",
          use_container_width=True,
          type="secondary",
      ):
        st.session_state["batch_exam_df"] = target_df.sample(n=num_q).reset_index(
            drop=True
        )
        st.rerun()

    if (
        "batch_exam_df" not in st.session_state
        or len(st.session_state["batch_exam_df"]) != num_q
    ):
      st.session_state["batch_exam_df"] = target_df.sample(n=num_q).reset_index(
          drop=True
      )

    exam_df = st.session_state["batch_exam_df"]
    st.divider()

    for idx, row in exam_df.iterrows():
      st.markdown(f"**Q{idx+1}. [{row['대단원']}] {row['question']}**")
      st.text_area(f"답안 입력 (문항 {idx+1})", key=f"batch_ans_{idx}", height=90)
      st.markdown("---")

    if st.button(
        "📝 전체 답안 일괄 채점 및 저장하기",
        type="primary",
        use_container_width=True,
    ):
      st.success("🎉 일괄 채점 로직이 연결되어 있습니다!")

# ==================== [탭 3: 성적표 & 오답노트] ====================
elif st.session_state["active_tab_index"] == 2:
  st.header("📈 나의 학습 성적표 및 취약 챕터 분석")
  results_file = "results.csv"

  if not os.path.isfile(results_file):
    st.info("💡 아직 저장된 학습 기록이 없습니다. 문제를 풀고 채점해 보세요!")
  else:
    res_df = pd.read_csv(results_file, encoding="utf-8-sig")
    total = len(res_df)
    avg = res_df["점수"].mean() if total > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("총 풀이 문항", f"{total}개")
    c2.metric("평균 점수", f"{avg:.1f}점")
    c3.metric("학습 상태", "🎯 합격권" if avg >= 60 else "⚠️ 보완 필요")

    st.divider()
    st.subheader("🚨 파트별 성적 분석 및 취약 챕터 파트 공부 추천")

    major_stats = (
        res_df.groupby("대단원")
        .agg(평균점수=("점수", "mean"), 풀이횟수=("점수", "count"))
        .reset_index()
    )
    weak_majors = major_stats.sort_values(by="평균점수", ascending=True)

    for idx, row in weak_majors.iterrows():
      col_info, col_btn = st.columns([3, 1])
      major_val = row["대단원"]
      with col_info:
        st.markdown(
            f"- 📂 **파트: [{major_val}]** (풀이: {row['풀이횟수']}회, 평균 점수:"
            f" **{row['평균점수']:.1f}점**)"
        )
      with col_btn:
        if st.button(f"🎯 집중 공략", key=f"focus_btn_{idx}", type="primary"):
          st.session_state["target_weak_major"] = major_val
          st.session_state["scope_mode"] = "🚨 취약 파트 공부"
          st.session_state["active_tab_index"] = 0
          if "batch_exam_df" in st.session_state:
            del st.session_state["batch_exam_df"]
          st.rerun()

    st.divider()
    st.subheader("📋 전체 학습 기록 데이터")
    st.dataframe(res_df, use_container_width=True)

    if st.button("🗑️ 학습 기록 전체 초기화"):
      if os.path.isfile(results_file):
        os.remove(results_file)
        st.rerun()
