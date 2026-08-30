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
  passimport os
import re
import csv
import chromadb
import pandas as pd
import streamlit as st
from google import genai

# 1. 페이지 설정 (가장 첫 줄 필수)
try:
  st.set_page_config(page_title="건축기사 RAG AI 학습 시스템", layout="wide")
except Exception:
  pass

# 2. Streamlit Secrets에서 Gemini API 키 안전하게 불러오기
try:
  GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
  st.error("⚠️ Streamlit Secrets에 'GEMINI_API_KEY'가 설정되지 않았습니다. 설정 후 다시 시도해주세요.")
  st.stop()

# 최신 구글 제미나이 SDK 클라이언트 초기화
client = genai.Client(api_key=GEMINI_API_KEY)

# 3. ChromaDB 영구 저장소 설정 (서버에 데이터가 날아가지 않고 저장됨)
@st.cache_resource
def init_chroma():
  return chromadb.PersistentClient(path="./chroma_db")

chroma_client = init_chroma()
collection = chroma_client.get_or_create_collection(name="architectural_exam_collection")

# 4. Excel 데이터 로드 및 RAG 벡터 DB 적재 함수 (이미지 전처리 및 대단원 재분류 포함)
@st.cache_data
def load_data():
  try:
    raw_df = pd.read_excel('data.xlsx', engine='openpyxl')
  except FileNotFoundError:
    # 파일이 없을 경우를 대비한 기본 셈플 데이터프레임 반환
    return pd.DataFrame(columns=['대단원', '중단원', '년도', '문제 내용', '모범 답안', '해설', '이미지'])
  
  raw_df.columns = raw_df.columns.str.strip()
  
  def clean_val(val):
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if s.lower() == 'nan':
        return ""
    return s

  processed_rows = []
  i = 0
  while i < len(raw_df):
    row = raw_df.iloc[i]
    q_text = clean_val(row.get('문제 내용'))
    major = clean_val(row.get('대단원'))
    
    if major == "" and q_text == "":
        i += 1
        continue
        
    middle = clean_val(row.get('중단원'))
    year = clean_val(row.get('년도'))
    correct = clean_val(row.get('모범 답안'))
    explanation = clean_val(row.get('해설'))
    
    img_path = None
    if i + 1 < len(raw_df):
        next_row = raw_df.iloc[i + 1]
        next_q_text = clean_val(next_row.get('문제 내용'))
        next_major = clean_val(next_row.get('대단원'))
        next_middle = clean_val(next_row.get('중단원'))
        
        if next_major == "" and next_middle == "" and \
           any(ext in next_q_text.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', 'images/']):
            img_path = next_q_text
            i += 1
            
    processed_rows.append({
        '대단원': major,
        '중단원': middle,
        '년도': year,
        '문제 내용': q_text,
        '모범 답안': correct,
        '해설': explanation,
        '이미지': img_path
    })
    i += 1

  df = pd.DataFrame(processed_rows)

  # 4대 대단원 자동 재분류 적용
  def reclassify_app_units(row):
    text = str(row['문제 내용'])
    old_major = str(row['대단원'])
    old_middle = str(row['중단원'])
    combined = old_major + " " + old_middle + " " + text
    
    if any(k in combined for k in ['공정', '네트워크', 'CPM', '공정표', 'VE', '가치공학', '선행작업', '후행작업']):
        return '공정관리'
    elif any(k in combined for k in ['적산', '견적', '수량', '단가', '공사비', '물량산출']):
        return '건축적산'
    elif any(k in combined for k in ['구조역학', '모멘트', '단면2차', '응력', '보의', '하중', '처짐', '철근콘크리트 구조', '철골구조', '내진', '휨모멘트']):
        return '건축구조'
    else:
        return '건축시공'

  df['대단원'] = df.apply(reclassify_app_units, axis=1)
  return df

df = load_data()

# ChromaDB에 데이터 적재 (최초 1회 또는 비어있을 때)
def vectorize_data_to_chroma(dataframe):
  if collection.count() == 0 and not dataframe.empty:
    documents = (dataframe['문제 내용'] + " " + dataframe['모범 답안'] + " " + dataframe['해설']).tolist()
    ids = [str(i) for i in range(len(dataframe))]
    metadatas = dataframe.fillna("").to_dict(orient="records")
    collection.add(documents=documents, ids=ids, metadatas=metadatas)

vectorize_data_to_chroma(df)

# 점수 추출 헬퍼 함수
def extract_score(result_text):
    match = re.search(r'(?:최종\s*점수|점수)[\s:]*([0-9]{1,3})점?', result_text)
    if match:
        return int(match.group(1))
    match_any = re.search(r'\b([0-9]{1,3})\b', result_text)
    if match_any:
        val = int(match_any.group(1))
        if 0 <= val <= 100:
            return val
    return 0

# ==================== [세션 상태 초기화] ====================
if 'scope_mode' not in st.session_state:
    st.session_state['scope_mode'] = "🎲 전체 챕터"
if 'target_weak_major' not in st.session_state:
    st.session_state['target_weak_major'] = None
if 'selected_major_val' not in st.session_state:
    major_unique = df['대단원'].unique().tolist() if not df.empty else []
    st.session_state['selected_major_val'] = major_unique[0] if major_unique else ""
if 'active_tab_index' not in st.session_state:
    st.session_state['active_tab_index'] = 0

# ==================== [사이드바: 학습 범위 설정] ====================
st.sidebar.markdown("### 🎛️ 공부할 범위 고르기")
current_mode = st.session_state['scope_mode']
st.sidebar.markdown(f"현재 학습 모드: **{current_mode}**")

if st.sidebar.button("🎲 전체 챕터로 변경", use_container_width=True):
    st.session_state['scope_mode'] = "🎲 전체 챕터"
    st.session_state['target_weak_major'] = None
    if 'batch_exam_df' in st.session_state:
        del st.session_state['batch_exam_df']
    st.rerun()

major_list = df['대단원'].unique().tolist() if not df.empty else ['건축시공', '공정관리', '건축적산', '건축구조']
selected_major_sb = st.sidebar.selectbox("📚 챕터별 학습 (대단원 선택)", major_list)
if st.sidebar.button("📚 선택한 챕터로 공부 시작", use_container_width=True):
    st.session_state['scope_mode'] = "📚 챕터별 학습"
    st.session_state['selected_major_val'] = selected_major_sb
    st.session_state['target_weak_major'] = None
    if 'batch_exam_df' in st.session_state:
        del st.session_state['batch_exam_df']
    st.rerun()

if st.session_state['scope_mode'] == "🚨 취약 파트 공부" and st.session_state['target_weak_major']:
    st.sidebar.warning(f"🚨 **집중 공략 중인 파트:**\n\n**{st.session_state['target_weak_major']}**")

st.sidebar.divider()

# 대상 데이터프레임 필터링
target_df = pd.DataFrame()
if df.empty:
    target_df = pd.DataFrame(columns=['대단원', '중단원', '년도', '문제 내용', '모범 답안', '해설', '이미지'])
elif st.session_state['scope_mode'] == "🎲 전체 챕터":
    target_df = df
elif st.session_state['scope_mode'] == "📚 챕터별 학습":
    target_df = df[df['대단원'] == st.session_state['selected_major_val']]
elif st.session_state['scope_mode'] == "🚨 취약 파트 공부":
    weak_m = st.session_state['target_weak_major']
    target_df = df[df['대단원'] == weak_m] if weak_m else df

# ==================== [메인 화면 타이틀 및 탭 네비게이션] ====================
st.title("🏗️ 건축기사 RAG AI 학습 & 채점 시스템")
st.markdown("딥러닝 임베딩과 **RAG(검색 증강 생성)** 기술을 결합하여 교재 내용을 기반으로 정확하게 정답을 찾고 서술형 답안을 정밀 채점합니다.")

col_t1, col_t2, col_t3 = st.columns(3)
with col_t1:
    if st.button("🎯 1단계: 문제 풀기 & RAG AI", use_container_width=True, type="primary" if st.session_state['active_tab_index']==0 else "secondary"):
        st.session_state['active_tab_index'] = 0
        st.rerun()
with col_t2:
    if st.button("📑 2단계: 시험지 모드", use_container_width=True, type="primary" if st.session_state['active_tab_index']==1 else "secondary"):
        st.session_state['active_tab_index'] = 1
        st.rerun()
with col_t3:
    if st.button("📊 3단계: 성적표 & 오답노트", use_container_width=True, type="primary" if st.session_state['active_tab_index']==2 else "secondary"):
        st.session_state['active_tab_index'] = 2
        st.rerun()

st.divider()

# 이미지 출력 헬퍼 함수
def render_question_image(row_data):
    img_path = row_data.get('이미지')
    if img_path and str(img_path).strip() != "":
        path_str = str(img_path).strip()
        if os.path.exists(path_str):
            st.image(path_str, caption="[문제 참고 그림]", width=500)
        else:
            try:
                st.image(path_str, caption="[문제 참고 그림]", width=500)
            except Exception:
                pass

# ==================== [탭 1: 한 문제씩 풀기 + RAG 검증 및 추가 질문] ====================
if st.session_state['active_tab_index'] == 0:
    if st.session_state['scope_mode'] == "🚨 취약 파트 공부":
        st.info(f"🚨 현재 **[{st.session_state['target_weak_major']}]** 파트 집중 공략 모드입니다!")
    else:
        st.markdown("#### 💡 한 문제씩 풀면서 RAG로 교재를 검색하고 즉시 채점 및 추가 질의를 할 수 있는 모드입니다.")
    
    q_list = target_df['문제 내용'].tolist() if not target_df.empty else []
    if not q_list:
        st.warning("⚠️ 'data.xlsx' 파일이 없거나 선택된 범위에 문제가 없습니다. data.xlsx 파일을 프로젝트 폴더에 넣어주세요.")
    else:
        selected_q = st.selectbox("📌 풀고 싶은 문제를 선택하세요:", q_list, key="single_q_select")
        row_data = target_df[target_df['문제 내용'] == selected_q].iloc[0]
        
        correct_answer = row_data['모범 답안']
        explanation = row_data['해설']
        question_year = row_data.get('년도', '정보 없음')
        q_major = row_data['대단원']
        q_sub = row_data['중단원']

        st.info(f"**[출제정보] 연도: {question_year}  |  대단원: {q_major}  |  중단원: {q_sub}**\n\n{selected_q}")
        render_question_image(row_data)

        user_ans = st.text_area("✍️ 정답을 서술형으로 입력하세요:", height=120, key="single_user_ans")

        if st.button("🤖 RAG 기반 AI 채점 요청하기", type="primary"):
            if not user_ans:
                st.warning("답안을 입력해주세요!")
            else:
                with st.spinner("🔍 RAG 벡터 검색 및 정밀 채점 진행 중..."):
                    # [RAG 검색 단계] 질문과 가장 유사한 교재 데이터를 Vector DB에서 검색
                    search_results = collection.query(query_texts=[selected_q], n_results=1)
                    retrieved_context = search_results["documents"][0][0] if search_results["documents"] else "관련 정보 없음"

                    # [루브릭 기반 프롬프트]
                    prompt = f"""
                    너는 엄격하고 공정한 건축기사 실기 수석 채점관이야.
                    아래의 [RAG 검색 교재 참고 내용]을 바탕으로 [학생의 답안]을 평가하고 점수를 매겨줘.

                    [RAG 검색 교재 내용]:
                    {retrieved_context}

                    [출제 문제]: {selected_q}
                    [모범 답안]: {correct_answer}
                    [상세 해설]: {explanation}
                    [학생 답안]: {user_ans}
                    
                    * 주의: 수식을 쓸 때 \\times, \\text 같은 LaTeX 문법은 절대 쓰지 말고 x, *, m^3 형태의 일반 텍스트만 사용하세요.
                    
                    핵심 키워드 포함 여부를 엄격히 평가하여 0~100점의 점수를 부여하고 피드백을 작성할 것.
                    반드시 아래 형식으로 출력할 것:
                    1. 최종 점수: XX점
                    2. 키워드 포함 여부: (...)
                    3. 채점 상세 평가: (...)
                    """
                    
                    response = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=prompt,
                    )
                    result_text = response.text
                    score = extract_score(result_text)

                    st.session_state['last_graded'] = {
                        "question": selected_q,
                        "user_ans": user_ans,
                        "score": score,
                        "result_text": result_text,
                        "correct_answer": correct_answer,
                        "explanation": explanation,
                        "retrieved_context": retrieved_context
                    }
                    st.session_state['messages'] = []

                    # CSV 결과 저장
                    file_name = 'results.csv'
                    file_exists = os.path.isfile(file_name)
                    with open(file_name, mode='a', newline='', encoding='utf-8-sig') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(['선택한문제', '대단원', '중단원', '년도', '학생답안', '점수', 'AI채점결과'])
                        writer.writerow([selected_q, q_major, q_sub, question_year, user_ans, score, result_text.replace('\n', ' ')])
                    st.success("채점 완료 및 오답노트 저장 완료!")

        if 'last_graded' in st.session_state and st.session_state['last_graded']['question'] == selected_q:
            lg = st.session_state['last_graded']
            st.markdown("---")
            st.markdown("### 📊 RAG 채점 결과 및 정답 확인")
            st.info(f"**점수: {lg['score']}점**")
            st.markdown(lg['result_text'])
            
            st.success(f"**📖 모범 답안**\n\n{lg['correct_answer']}")
            st.info(f"**💡 상세 해설**\n\n{lg['explanation']}")
            
            with st.expander("🔍 [RAG 시스템] 검색된 교재 원문 확인하기"):
                st.write(lg['retrieved_context'])
            st.markdown("---")

        st.markdown("##### 💬 AI에게 이어서 질문하기")
        if 'messages' not in st.session_state:
            st.session_state['messages'] = []

        for message in st.session_state['messages']:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if chat_input := st.chat_input("예: 이 개념이 실기 시험에 또 어떻게 응용돼서 나와?"):
            st.session_state['messages'].append({"role": "user", "content": chat_input})
            with st.chat_message("user"):
                st.markdown(chat_input)

            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    chat_history_text = f"너는 건축기사 수석 강사야. 현재 문제: '{selected_q}', 모범답안: '{correct_answer}', 해설: '{explanation}'\n 학생 질문: {chat_input}"
                    chat_response = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=chat_history_text
                    )
                    answer_text = chat_response.text
                    st.markdown(answer_text)
                    st.session_state['messages'].append({"role": "assistant", "content": answer_text})

# ==================== [탭 2: 시험지 모드] ====================
elif st.session_state['active_tab_index'] == 1:
    st.markdown("#### 📑 지정한 문항 수만큼 무작위로 시험지를 구성하여 한 번에 풀고 RAG 일괄 채점을 받는 모드입니다.")
    
    if target_df.empty:
        st.warning("⚠️ 선택된 범위에 문제가 없습니다.")
    else:
        c_cnt, c_action = st.columns([2, 2])
        with c_cnt:
            max_limit = len(target_df)
            num_q = st.number_input("추출 문항 수 설정", min_value=1, max_value=max(1, max_limit), value=min(5, max_limit), key="batch_num_q")
        with c_action:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🎲 새로운 문제 세트 무작위 뽑기", use_container_width=True, type="secondary"):
                st.session_state['batch_exam_df'] = target_df.sample(n=num_q).reset_index(drop=True)
                st.rerun()

        if 'batch_exam_df' not in st.session_state or len(st.session_state['batch_exam_df']) != num_q:
            st.session_state['batch_exam_df'] = target_df.sample(n=num_q).reset_index(drop=True)

        exam_df = st.session_state['batch_exam_df']
        st.divider()

        user_answers_dict = {}
        for idx, row in exam_df.iterrows():
            q_year = row.get('년도', '정보 없음')
            st.markdown(f"**Q{idx+1}. [{q_year} | {row['대단원']} > {row['중단원']}] {row['문제 내용']}**")
            render_question_image(row)

            ans = st.text_area(f"답안 입력 (문항 {idx+1})", key=f"batch_ans_{idx}", height=90)
            user_answers_dict[idx] = {
                "question": row['문제 내용'],
                "major": row['대단원'],
                "sub": row['중단원'],
                "year": q_year,
                "correct": row['모범 답안'],
                "explanation": row['해설'],
                "user_ans": ans
            }
            st.markdown("---")

        if st.button("📝 전체 답안 RAG 일괄 채점 및 저장하기", type="primary", use_container_width=True):
            with st.spinner("🤖 RAG 검색 및 일괄 채점 진행 중..."):
                file_name = 'results.csv'
                file_exists = os.path.isfile(file_name)
                batch_results = []

                for idx, data in user_answers_dict.items():
                    if not data["user_ans"]:
                        continue
                    
                    search_results = collection.query(query_texts=[data['question']], n_results=1)
                    retrieved_context = search_results["documents"][0][0] if search_results["documents"] else "관련 정보 없음"

                    prompt = f"""
                    너는 건축기사 실기 수석 채점관이야.
                    [RAG 교재 참고 내용]: {retrieved_context}
                    [문제]: {data['question']}
                    [모범 답안]: {data['correct']}
                    [학생 답안]: {data['user_ans']}
                    
                    핵심 키워드 포함 여부를 엄격하게 평가하여 0~100점의 점수를 부여하고 피드백을 줘.
                    반드시 아래 형식으로 출력할 것:
                    1. 최종 점수: XX점
                    2. 피드백: (...)
                    """
                    
                    try:
                        response = client.models.generate_content(
                            model="gemini-3.6-flash",
                            contents=prompt
                        )
                        res_text = response.text
                        score = extract_score(res_text)
                    except Exception as e:
                        res_text = f"채점 오류: {str(e)}"
                        score = 0
                        
                    batch_results.append({
                        "question": data['question'], 
                        "user_ans": data['user_ans'], 
                        "score": score, 
                        "result": res_text,
                        "correct": data['correct'],
                        "explanation": data['explanation']
                    })
                    
                    with open(file_name, mode='a', newline='', encoding='utf-8-sig') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(['선택한문제', '대단원', '중단원', '년도', '학생답안', '점수', 'AI채점결과'])
                            file_exists = True
                        writer.writerow([data['question'], data['major'], data['sub'], data['year'], data['user_ans'], score, res_text.replace('\n', ' ')])

                st.success("🎉 일괄 채점이 완료되었습니다!")
                for res in batch_results:
                    with st.expander(f"📌 [점수: {res['score']}점] {res['question'][:35]}..."):
                        st.markdown(f"**내 답안:** {res['user_ans']}")
                        st.markdown(f"**채점 결과:**\n{res['result']}")
                        st.markdown(f"**[모범 답안]**\n{res['correct']}")
                        st.markdown(f"**[상세 해설]**\n{res['explanation']}")

# ==================== [탭 3: 학습 분석 & 오답노트] ====================
elif st.session_state['active_tab_index'] == 2:
    st.header("📈 나의 학습 성적표 및 취약 챕터 분석")
    results_file = 'results.csv'
    
    if not os.path.isfile(results_file):
        st.info("💡 아직 저장된 학습 기록이 없습니다. 문제를 풀고 채점해 보세요!")
    else:
        res_df = pd.read_csv(results_file, encoding='utf-8-sig')
        if '대단원' not in res_df.columns:
            res_df['대단원'] = '건축시공'

        total = len(res_df)
        avg = res_df['점수'].mean() if total > 0 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("총 풀이 문항", f"{total}개")
        c2.metric("평균 점수", f"{avg:.1f}점")
        c3.metric("학습 상태", "🎯 합격권" if avg >= 60 else "⚠️ 보완 필요")
        
        st.divider()
        st.subheader("🚨 파트별 성적 분석 및 취약 파트 집중 공략 추천")
        major_stats = res_df.groupby('대단원').agg(
            평균점수=('점수', 'mean'),
            풀이횟수=('점수', 'count')
        ).reset_index()
        
        weak_majors = major_stats.sort_values(by='평균점수', ascending=True)
        
        if not weak_majors.empty:
            st.markdown("👇 점수가 낮게 나온 파트의 **[🎯 집중 공략]** 버튼을 누르면 해당 파트만 집중 학습할 수 있습니다!")
            for idx, row in weak_majors.head(5).iterrows():
                major_name = row['대단원']
                avg_s = row['평균점수']
                count = row['풀이횟수']
                
                col_info, col_btn = st.columns([3, 1])
                with col_info:
                    st.markdown(f"- 📂 **파트: [{major_name}]** (풀이: {count}회, 평균 점수: **{avg_s:.1f}점**)")
                with col_btn:
                    if st.button(f"🎯 집중 공략", key=f"focus_btn_{idx}", type="primary"):
                        st.session_state['target_weak_major'] = major_name
                        st.session_state['scope_mode'] = "🚨 취약 파트 공부"
                        st.session_state['active_tab_index'] = 0 
                        if 'batch_exam_df' in st.session_state:
                            del st.session_state['batch_exam_df']
                        st.rerun()
        
        st.divider()
        st.subheader("📋 전체 학습 기록 데이터")
        st.dataframe(res_df, use_container_width=True)
        
        if st.button("🗑️ 학습 기록 전체 초기화"):
            if os.path.isfile(results_file):
                os.remove(results_file)
                st.rerun()

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
