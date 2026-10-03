import io
import json
import os
import random
import re
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
from docx import Document
from PIL import Image, ImageDraw, ImageFont


# =========================
# 기본 설정
# =========================
st.set_page_config(
    page_title="소설 속 인물 인터뷰",
    page_icon="📚",
    layout="centered",
)

MODEL_NAME = "gemini-3.8-flash"

def get_secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


ADMIN_PASSWORD = str(get_secret("ADMIN_PASSWORD", "6460"))
DATA_DIR = Path("data")
WORKS_FILE = DATA_DIR / "works.json"
MAX_TURNS = 5
MAX_CONTEXT_CHARS = 60000

DATA_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 데이터 저장/불러오기
# =========================
def load_works():
    if not WORKS_FILE.exists():
        return []
    try:
        with open(WORKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_works(works):
    with open(WORKS_FILE, "w", encoding="utf-8") as f:
        json.dump(works, f, ensure_ascii=False, indent=2)


def get_work_by_id(work_id):
    for work in load_works():
        if work.get("id") == work_id:
            return work
    return None


# =========================
# Gemini 연결
# =========================
def get_api_key():
    key = get_secret("GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY")
    return key


@st.cache_resource
def get_client(api_key):
    return genai.Client(api_key=api_key)


def generate_text(prompt, use_google_search=False, temperature=0.4):
    """
    Google Search grounding을 우선 시도하고, 무료 등급 등에서 검색 도구를 쓸 수 없으면
    검색 없이 Gemini 자체 지식으로 한 번 더 시도한다.
    반환값: (텍스트, 실제 웹검색 사용 여부, 검색 실패 메시지)
    """
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 없습니다.")

    client = get_client(api_key)

    config_kwargs = {
        "temperature": temperature,
    }

    if use_google_search:
        try:
            config = types.GenerateContentConfig(
                **config_kwargs,
                tools=[types.Tool(google_search=types.GoogleSearch())],
            )
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config,
            )
            return (response.text or "").strip(), True, None
        except Exception as e:
            search_error = str(e)
            # 무료 API 키처럼 Google Search 도구를 쓸 수 없는 경우에도 앱이 멈추지 않게 재시도
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            return (response.text or "").strip(), False, search_error

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    return (response.text or "").strip(), False, None


# =========================
# 업로드 파일 텍스트 추출
# =========================
def decode_txt(data: bytes):
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def extract_hwpx_text(data: bytes):
    texts = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        section_names = sorted(
            [n for n in zf.namelist() if n.lower().startswith("contents/section") and n.lower().endswith(".xml")]
        )
        for name in section_names:
            root = ET.fromstring(zf.read(name))
            for elem in root.iter():
                # HWPX의 텍스트 노드는 보통 네임스페이스를 포함한 t 태그
                if elem.tag.split("}")[-1] == "t" and elem.text:
                    texts.append(elem.text)
    return "\n".join(texts)


def extract_uploaded_text(uploaded_file):
    if uploaded_file is None:
        return ""

    name = uploaded_file.name
    suffix = Path(name).suffix.lower()
    data = uploaded_file.getvalue()

    if suffix == ".txt":
        text = decode_txt(data)
    elif suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    elif suffix == ".docx":
        doc = Document(io.BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs)
    elif suffix == ".hwpx":
        text = extract_hwpx_text(data)
    else:
        raise ValueError("지원하지 않는 파일 형식입니다.")

    return re.sub(r"\n{3,}", "\n\n", text).strip()


def files_to_sources(uploaded_files):
    sources = []
    errors = []
    for file in uploaded_files or []:
        try:
            text = extract_uploaded_text(file)
            if not text.strip():
                errors.append(f"{file.name}: 읽을 수 있는 텍스트가 없습니다. 스캔 PDF라면 텍스트 PDF로 변환해 주세요.")
                continue
            sources.append({"name": file.name, "text": text})
        except Exception as e:
            errors.append(f"{file.name}: {e}")
    return sources, errors


# =========================
# 문맥 추출
# =========================
def tokenize_korean(text):
    return set(re.findall(r"[가-힣A-Za-z0-9]{2,}", text or ""))


def chunk_text(text, chunk_size=1800, overlap=250):
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def select_relevant_context(work, question, character, max_chars=MAX_CONTEXT_CHARS):
    source_blocks = []
    for label, key in (("소설 전문/본문", "novel_sources"), ("학습자료", "learning_sources")):
        for src in work.get(key, []) or []:
            source_blocks.append((label, src.get("name", "자료"), src.get("text", "")))

    if not source_blocks:
        return ""

    query_terms = tokenize_korean(f"{question} {character} {work.get('title','')} {work.get('author','')}")
    scored = []
    for label, name, text in source_blocks:
        for i, chunk in enumerate(chunk_text(text)):
            terms = tokenize_korean(chunk)
            score = len(query_terms & terms)
            if character and character in chunk:
                score += 8
            scored.append((score, label, name, i, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = []
    total = 0
    seen = set()

    # 관련도가 높은 부분 우선
    for score, label, name, i, chunk in scored:
        key = (label, name, i)
        if key in seen:
            continue
        block = f"[{label} - {name}]\n{chunk}"
        if total + len(block) > max_chars:
            continue
        chosen.append(block)
        seen.add(key)
        total += len(block)
        if total >= max_chars * 0.8:
            break

    # 질문어가 짧아 관련도 점수가 거의 없을 때 자료 앞부분도 보강
    if len(chosen) < 4:
        for label, name, text in source_blocks:
            block = f"[{label} - {name}]\n{text[:5000]}"
            if block not in chosen and total + len(block) <= max_chars:
                chosen.append(block)
                total += len(block)

    return "\n\n".join(chosen)


# =========================
# 등장인물 자동 찾기
# =========================
def parse_character_names(text):
    text = text.strip()
    # 코드펜스 제거
    text = re.sub(r"```(?:json)?", "", text, flags=re.I).replace("```", "").strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            names = obj.get("characters", [])
            if isinstance(names, list):
                return [str(x).strip() for x in names if str(x).strip()][:10]
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()][:10]
    except Exception:
        pass

    parts = re.split(r"[,\n/·•]+", text)
    cleaned = []
    for p in parts:
        p = re.sub(r"^\s*[-\d.()]+\s*", "", p).strip()
        if p and len(p) <= 20 and p not in cleaned:
            cleaned.append(p)
    return cleaned[:10]


def discover_characters(work):
    title = work.get("title", "")
    author = work.get("author", "")
    context = select_relevant_context(work, "주요 등장인물 이름", "")
    has_novel = bool(work.get("novel_sources"))

    prompt = f"""
너는 중학교 국어 수업용 작품 분석 도우미다.
작품명: {title}
작가: {author}

아래 자료가 있으면 가장 우선해서 확인하고, 자료가 부족하면 작품명과 작가를 기준으로 작품을 식별하라.
동명이인이나 동명 작품을 섞지 마라.
학생이 인터뷰하기 적절한 '실제 등장인물 이름'을 2~8명 골라라.
인물 이름을 확신할 수 없으면 추측해서 만들지 마라.

자료:
{context if context else '(교사가 파일을 올리지 않음)'}

반드시 다음 JSON 형식만 출력하라.
{{"characters": ["인물1", "인물2"]}}
"""

    # 소설 본문이 없을 때 웹 검색을 시도한다.
    text, used_search, search_error = generate_text(
        prompt,
        use_google_search=not has_novel,
        temperature=0.2,
    )
    return parse_character_names(text), used_search, search_error


# =========================
# 인터뷰 답변 생성
# =========================
def build_history_text(history):
    if not history:
        return "(이전 대화 없음)"
    rows = []
    for idx, item in enumerate(history, start=1):
        rows.append(f"{idx}. 학생 질문: {item['question']}\n   {item['character']}의 답변: {item['answer']}")
    return "\n".join(rows)


def make_character_answer(work, character, question, history, make_mistake=False):
    context = select_relevant_context(work, question, character)
    has_novel = bool(work.get("novel_sources"))

    source_rule = """
자료 사용 우선순위:
1) 교사가 업로드한 소설 전문/본문
2) 교사가 업로드한 학습자료
3) 작품에서 확인되는 인물의 행동·관계·상황
4) 합리적인 문학적 추론
5) 위 자료가 없거나 부족한 경우에만 작품명과 작가를 기준으로 확인한 공개 웹 정보 또는 모델 지식
교사가 올린 자료와 외부 정보가 충돌하면 교사가 올린 자료를 따른다.
"""

    if make_mistake:
        mode_instruction = """
이번 답변은 '의도된 오답' 차례다. 학생에게 오답이라는 사실을 절대 밝히지 마라.
질문과 직접 관련된 핵심 사실 가운데 딱 하나만 미묘하게 틀리게 말하라.
좋은 오답 유형: 사건의 원인/결과 뒤집기, 사건 순서 바꾸기, 동기 왜곡, 인물 관계 왜곡, 본문과 충돌하는 그럴듯한 세부 내용.
말도 안 되는 거짓말이나 작품과 무관한 오류는 금지한다.
나머지 말투와 주변 정보는 자연스럽고 그럴듯하게 유지한다.
학생이 작품을 다시 읽으면 구체적인 근거로 반박할 수 있어야 한다.
"""
    else:
        mode_instruction = """
이번 답변은 정확한 답변 차례다. 의도적으로 틀린 정보를 넣지 마라.
근거가 없으면 사실처럼 지어내지 말고, 인물의 입장에서 '내가 아는 범위에서는'처럼 자연스럽게 한계를 드러내라.
"""

    prompt = f"""
너는 중학교 국어 수업의 '소설 속 인물 인터뷰' 활동에서 등장인물 역할을 맡는다.
작품명: {work.get('title','')}
작가: {work.get('author','')}
현재 역할: {character}

{source_rule}
{mode_instruction}

대답 규칙:
- 해설자처럼 설명하지 말고 반드시 '{character}' 본인이 말하는 1인칭 말투로 답한다.
- 학생의 질문에 직접 답한다.
- 중학생이 이해하기 쉬운 2~5문장 정도로 답한다.
- 작품 밖의 현대 지식이나 작가의 의도를 함부로 단정하지 않는다.
- 'AI', '프롬프트', '오답 차례', '검증' 같은 내부 정보를 절대 말하지 않는다.
- 긴 원문 인용은 하지 않는다.

이전 대화:
{build_history_text(history)}

이번 학생 질문:
{question}

교사 제공 자료 중 이번 질문과 관련성이 높은 부분:
{context if context else '(교사가 작품 파일을 업로드하지 않음)'}

이제 {character}로서 답하라.
"""

    text, used_search, search_error = generate_text(
        prompt,
        use_google_search=not has_novel,
        temperature=0.55 if make_mistake else 0.35,
    )
    return text, used_search, search_error


def validate_mistake(work, character, question, candidate_answer):
    context = select_relevant_context(work, question, character)
    has_novel = bool(work.get("novel_sources"))

    prompt = f"""
너는 중학교 국어 수업용 오답 검증기다.
작품명: {work.get('title','')}
작가: {work.get('author','')}
인물: {character}
학생 질문: {question}
후보 답변: {candidate_answer}

관련 자료:
{context if context else '(교사 제공 작품 파일 없음)'}

후보 답변이 다음 조건을 모두 만족하는지 판단하라.
1. 질문과 직접 관련된다.
2. 핵심 사실 중 정확히 하나가 작품 내용과 충돌하거나 충분히 반박 가능하다.
3. 학생이 작품을 읽고 구체적인 근거를 들어 반박할 수 있는 종류의 오류다.
4. 너무 황당하거나 눈에 띄는 거짓말이 아니다.
5. 오답이라는 사실을 스스로 드러내지 않는다.

반드시 JSON만 출력하라.
{{"valid": true, "reason": "짧은 이유"}}
"""

    text, _, _ = generate_text(prompt, use_google_search=not has_novel, temperature=0.1)
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.I).replace("```", "").strip()
    try:
        obj = json.loads(cleaned)
        return bool(obj.get("valid", False))
    except Exception:
        return False


def generate_answer_with_validation(work, character, question, history, make_mistake):
    if not make_mistake:
        return make_character_answer(work, character, question, history, make_mistake=False)

    last = None
    for _ in range(3):
        answer, used_search, search_error = make_character_answer(
            work, character, question, history, make_mistake=True
        )
        last = (answer, used_search, search_error)
        try:
            if validate_mistake(work, character, question, answer):
                return last
        except Exception:
            # 검증 호출 자체가 실패해도 인터뷰 전체가 멈추지 않게 한다.
            return last
    return last


# =========================
# 결과 검토
# =========================
def evaluate_evidence(work, character, turn_item, evidence):
    context = select_relevant_context(work, turn_item["question"], character)
    has_novel = bool(work.get("novel_sources"))

    prompt = f"""
너는 중학교 국어 교사 보조자다.
작품명: {work.get('title','')}
작가: {work.get('author','')}
학생이 찾아낸 의도된 오답 질문: {turn_item['question']}
AI의 당시 답변: {turn_item['answer']}
학생이 제시한 작품 근거: {evidence}

관련 교사 자료:
{context if context else '(교사 제공 작품 파일 없음)'}

학생의 근거가 오답을 반박하는 데 적절한지 짧게 평가하라.
- 적절하면 왜 근거가 되는지 2~3문장으로 설명한다.
- 부족하면 어떤 부분을 다시 찾아야 하는지 힌트만 준다.
- 작품에 없는 내용을 만들어내지 않는다.
"""
    text, _, _ = generate_text(prompt, use_google_search=not has_novel, temperature=0.2)
    return text


# =========================
# 결과 이미지
# =========================
def load_font(size):
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def wrap_text(text, width=34):
    words = list(text)
    lines = []
    line = ""
    for ch in words:
        line += ch
        if len(line) >= width and ch in " ,.!?。！？,， ":
            lines.append(line.strip())
            line = ""
    if line.strip():
        lines.append(line.strip())
    return lines or [text]


def make_result_png(work, character, history, picked_turn, evidence):
    width = 1200
    base_height = 500 + len(history) * 220 + max(120, len(evidence) * 2)
    image = Image.new("RGB", (width, base_height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(38)
    body_font = load_font(26)
    small_font = load_font(22)

    y = 45
    draw.text((55, y), "소설 속 인물 인터뷰 결과", font=title_font, fill="black")
    y += 65
    draw.text((55, y), f"작품: {work.get('title','')} / {work.get('author','')}", font=body_font, fill="black")
    y += 45
    draw.text((55, y), f"인터뷰 인물: {character}", font=body_font, fill="black")
    y += 60

    for idx, item in enumerate(history, start=1):
        q_lines = wrap_text(f"Q{idx}. {item['question']}", 45)
        a_lines = wrap_text(f"A. {item['answer']}", 45)
        for line in q_lines + a_lines:
            draw.text((55, y), line, font=small_font, fill="black")
            y += 34
        y += 18

    draw.text((55, y), f"내가 고른 오답: {picked_turn}번", font=body_font, fill="black")
    y += 48
    for line in wrap_text(f"작품 근거: {evidence}", 45):
        draw.text((55, y), line, font=small_font, fill="black")
        y += 34

    output = io.BytesIO()
    image.crop((0, 0, width, min(base_height, y + 60))).save(output, format="PNG")
    output.seek(0)
    return output


# =========================
# 세션 상태
# =========================
def init_state():
    defaults = {
        "admin_ok": False,
        "interview_started": False,
        "active_work_id": None,
        "active_character": None,
        "history": [],
        "mistake_turn": None,
        "finished": False,
        "busy": False,
        "last_search_notice": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_interview():
    for key, value in {
        "interview_started": False,
        "active_work_id": None,
        "active_character": None,
        "history": [],
        "mistake_turn": None,
        "finished": False,
        "busy": False,
        "last_search_notice": None,
    }.items():
        st.session_state[key] = value


init_state()


# =========================
# 스타일
# =========================
st.markdown(
    """
    <style>
    .block-container {max-width: 900px; padding-top: 2rem; padding-bottom: 4rem;}
    .progress-box {padding: 0.65rem 0.9rem; border: 1px solid #ddd; border-radius: 12px; text-align:center; font-weight:700;}
    .source-note {font-size: 0.92rem; color: #666;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# 상단 메뉴
# =========================
page = st.sidebar.radio("메뉴", ["학생 화면", "교사 설정"])


# =========================
# 교사 설정 화면
# =========================
if page == "교사 설정":
    st.title("⚙️ 교사 설정")

    if not st.session_state.admin_ok:
        pw = st.text_input("관리자 비밀번호", type="password")
        if st.button("로그인", use_container_width=True):
            if pw == ADMIN_PASSWORD:
                st.session_state.admin_ok = True
                st.rerun()
            else:
                st.error("비밀번호가 맞지 않습니다.")
        st.stop()

    st.success("관리자 모드입니다.")
    st.caption("작품명과 작가만 필수입니다. 등장인물·소설 파일·학습자료 파일은 비워 두어도 저장할 수 있습니다.")

    works = load_works()
    option_labels = ["➕ 새 작품 만들기"] + [f"{w.get('title','')} — {w.get('author','')}" for w in works]
    selected_label = st.selectbox("작품 선택", option_labels)
    selected_index = option_labels.index(selected_label)
    editing = selected_index > 0
    current = works[selected_index - 1] if editing else None

    title = st.text_input("작품명 *", value=current.get("title", "") if current else "")
    author = st.text_input("작가 이름 *", value=current.get("author", "") if current else "")

    existing_chars = ", ".join(current.get("characters", [])) if current else ""
    chars_text = st.text_input(
        "등장인물 (선택)",
        value=existing_chars,
        placeholder="예: 아버지, 나, 노새  / 비워 두면 자동으로 찾습니다.",
    )

    st.subheader("소설 전문/본문 파일 (선택)")
    if current and current.get("novel_sources"):
        st.caption("현재 저장된 파일: " + ", ".join(x.get("name", "") for x in current["novel_sources"]))
    novel_files = st.file_uploader(
        "TXT, PDF, DOCX, HWPX 파일을 업로드하세요.",
        type=["txt", "pdf", "docx", "hwpx"],
        accept_multiple_files=True,
        key="novel_uploads",
    )
    clear_novel = st.checkbox("기존 소설 파일 내용 삭제", value=False) if editing else False

    st.subheader("학습자료 파일 (선택)")
    if current and current.get("learning_sources"):
        st.caption("현재 저장된 파일: " + ", ".join(x.get("name", "") for x in current["learning_sources"]))
    learning_files = st.file_uploader(
        "TXT, PDF, DOCX, HWPX 파일을 업로드하세요.",
        type=["txt", "pdf", "docx", "hwpx"],
        accept_multiple_files=True,
        key="learning_uploads",
    )
    clear_learning = st.checkbox("기존 학습자료 내용 삭제", value=False) if editing else False

    st.info(
        "파일을 올리지 않아도 저장됩니다. 다만 현재 Gemini 무료 API에서는 Google 검색 그라운딩이 제한될 수 있어, "
        "실제 웹 검색을 쓸 수 없는 경우 Gemini 자체 지식으로 자동 대체됩니다."
    )

    col_save, col_delete = st.columns([3, 1])
    with col_save:
        do_save = st.button("💾 저장", type="primary", use_container_width=True)
    with col_delete:
        do_delete = st.button("🗑️ 삭제", use_container_width=True, disabled=not editing)

    if do_delete and current:
        works = [w for w in works if w.get("id") != current.get("id")]
        save_works(works)
        st.success("작품을 삭제했습니다.")
        st.rerun()

    if do_save:
        if not title.strip() or not author.strip():
            st.error("작품명과 작가 이름은 반드시 입력해 주세요.")
            st.stop()

        new_novel_sources, novel_errors = files_to_sources(novel_files)
        new_learning_sources, learning_errors = files_to_sources(learning_files)
        for msg in novel_errors + learning_errors:
            st.warning(msg)

        if current:
            novel_sources = [] if clear_novel else list(current.get("novel_sources", []))
            learning_sources = [] if clear_learning else list(current.get("learning_sources", []))
        else:
            novel_sources = []
            learning_sources = []

        # 같은 이름의 새 파일을 올리면 기존 내용을 교체
        def merge_sources(existing, new_items):
            by_name = {x.get("name"): x for x in existing}
            for item in new_items:
                by_name[item.get("name")] = item
            return list(by_name.values())

        novel_sources = merge_sources(novel_sources, new_novel_sources)
        learning_sources = merge_sources(learning_sources, new_learning_sources)

        manual_chars = [x.strip() for x in re.split(r"[,/\n]+", chars_text) if x.strip()]

        work = {
            "id": current.get("id") if current else str(uuid.uuid4()),
            "title": title.strip(),
            "author": author.strip(),
            "characters": manual_chars,
            "novel_sources": novel_sources,
            "learning_sources": learning_sources,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

        # 등장인물을 비워 둔 경우 자동으로 찾아서 저장
        search_notice = None
        if not manual_chars:
            try:
                with st.spinner("등장인물을 자동으로 찾고 있어요..."):
                    auto_chars, used_search, search_error = discover_characters(work)
                if auto_chars:
                    work["characters"] = auto_chars
                    if used_search:
                        search_notice = "등장인물을 Google 검색을 통해 확인했습니다."
                    elif search_error:
                        search_notice = "Google 검색을 사용할 수 없어 Gemini 자체 지식으로 등장인물을 찾았습니다."
                else:
                    search_notice = "등장인물을 자동으로 찾지 못했습니다. 나중에 교사 설정에서 직접 입력할 수 있습니다."
            except Exception as e:
                search_notice = f"등장인물 자동 찾기를 건너뛰었습니다: {e}"

        if current:
            works = [work if w.get("id") == current.get("id") else w for w in works]
        else:
            works.append(work)
        save_works(works)

        st.success("저장했습니다.")
        if search_notice:
            st.info(search_notice)
        st.rerun()

    st.divider()
    st.caption("※ 이 프로토타입은 작품 정보를 Streamlit 서버의 로컬 JSON 파일에 저장합니다. 앱 재배포·재시작 시 영구 보존이 보장되지 않습니다.")


# =========================
# 학생 화면
# =========================
else:
    st.title("📚 소설 속 인물 인터뷰")
    st.caption("등장인물에게 5번 질문하고, 답변 속에 숨어 있는 한 번의 오류를 작품 근거로 찾아보세요.")

    if not get_api_key():
        st.error("GEMINI_API_KEY가 없습니다. Streamlit Cloud의 Secrets에 Gemini API 키를 등록해 주세요.")
        st.stop()

    works = load_works()
    if not works:
        st.info("교사 설정에서 작품을 먼저 등록해 주세요.")
        st.stop()

    if not st.session_state.interview_started:
        col1, col2 = st.columns([2, 2])
        with col1:
            work_labels = [f"{w.get('title','')} — {w.get('author','')}" for w in works]
            selected_work_label = st.selectbox("작품", work_labels)
            work = works[work_labels.index(selected_work_label)]

        # 등장인물이 비어 있으면 학생 화면에서도 한 번 더 자동 탐색
        characters = work.get("characters", [])
        if not characters:
            try:
                with st.spinner("등장인물을 불러오고 있어요..."):
                    characters, used_search, search_error = discover_characters(work)
                if characters:
                    work["characters"] = characters
                    works = [work if w.get("id") == work.get("id") else w for w in works]
                    save_works(works)
                if search_error and not used_search:
                    st.caption("웹 검색을 사용할 수 없어 Gemini 지식으로 등장인물을 불러왔습니다.")
            except Exception:
                characters = []

        with col2:
            if characters:
                character = st.selectbox("인터뷰할 인물", characters)
            else:
                character = st.text_input(
                    "인터뷰할 인물 이름",
                    placeholder="등장인물 자동 찾기에 실패했습니다. 이름을 직접 입력해 주세요.",
                )

        if st.button("인터뷰 시작", type="primary", use_container_width=True):
            if not character or not str(character).strip():
                st.warning("인터뷰할 인물을 선택하거나 입력해 주세요.")
            else:
                st.session_state.interview_started = True
                st.session_state.active_work_id = work.get("id")
                st.session_state.active_character = str(character).strip()
                st.session_state.history = []
                st.session_state.mistake_turn = random.randint(1, MAX_TURNS)
                st.session_state.finished = False
                st.rerun()
        st.stop()

    work = get_work_by_id(st.session_state.active_work_id)
    if not work:
        st.error("작품 정보를 찾을 수 없습니다. 인터뷰를 다시 시작해 주세요.")
        if st.button("처음으로"):
            reset_interview()
            st.rerun()
        st.stop()

    character = st.session_state.active_character
    turn_count = len(st.session_state.history)

    top_left, top_right = st.columns([3, 1])
    with top_left:
        st.subheader(f"{work.get('title','')} · {character}")
        st.caption(f"{work.get('author','')} 작품")
    with top_right:
        st.markdown(
            f'<div class="progress-box">질문 {min(turn_count + 1, MAX_TURNS)} / {MAX_TURNS}</div>',
            unsafe_allow_html=True,
        )

    dots = " ".join("●" if i <= turn_count else "○" for i in range(1, MAX_TURNS + 1))
    st.markdown(f"**진행:** {dots}")

    for idx, item in enumerate(st.session_state.history, start=1):
        with st.chat_message("user"):
            st.markdown(f"**Q{idx}.** {item['question']}")
        with st.chat_message("assistant"):
            st.markdown(item["answer"])

    if turn_count < MAX_TURNS and not st.session_state.finished:
        with st.form("question_form", clear_on_submit=True):
            question = st.text_input(
                "질문",
                max_chars=300,
                placeholder=f"{character}에게 궁금한 점을 한 가지 질문해 보세요.",
                disabled=st.session_state.busy,
            )
            submitted = st.form_submit_button(
                "질문하기",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.busy,
            )

        if submitted:
            q = (question or "").strip()
            if not q:
                st.warning("질문을 입력해 주세요.")
            else:
                st.session_state.busy = True
                current_turn = turn_count + 1
                make_mistake = current_turn == st.session_state.mistake_turn
                try:
                    with st.spinner(f"{character}가 답변을 생각하고 있어요..."):
                        answer, used_search, search_error = generate_answer_with_validation(
                            work,
                            character,
                            q,
                            st.session_state.history,
                            make_mistake,
                        )
                    if not answer:
                        raise RuntimeError("빈 답변이 생성되었습니다.")

                    st.session_state.history.append(
                        {
                            "question": q,
                            "answer": answer,
                            "character": character,
                        }
                    )

                    if search_error and not used_search:
                        st.session_state.last_search_notice = (
                            "교사 제공 소설 파일이 없어 웹 검색을 시도했지만, 현재 API 설정에서는 검색을 사용할 수 없어 "
                            "Gemini 자체 지식으로 답했습니다."
                        )

                    if len(st.session_state.history) >= MAX_TURNS:
                        st.session_state.finished = True
                    st.session_state.busy = False
                    st.rerun()
                except Exception as e:
                    st.session_state.busy = False
                    st.error(f"답변을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요. ({e})")

    if st.session_state.last_search_notice:
        st.caption("※ " + st.session_state.last_search_notice)

    if st.session_state.finished:
        st.divider()
        st.subheader("🔎 이제 AI의 오류를 찾아보세요")
        st.write("5개의 답변 중 작품 내용과 어긋나는 답변은 **딱 하나**입니다.")

        with st.form("result_form"):
            picked = st.selectbox("틀린 답변이라고 생각하는 번호", list(range(1, MAX_TURNS + 1)))
            evidence = st.text_area(
                "작품 속 근거",
                placeholder="왜 그 답변이 틀렸는지 작품의 사건, 행동, 대사 등을 근거로 적어 보세요.",
                height=130,
            )
            check = st.form_submit_button("확인하기", type="primary", use_container_width=True)

        if check:
            if not evidence.strip():
                st.warning("작품 속 근거를 적어 주세요.")
            elif picked == st.session_state.mistake_turn:
                st.success("오답의 위치를 정확히 찾았습니다! 이제 작품 근거가 적절한지 확인해 볼게요.")
                try:
                    with st.spinner("작품 근거를 확인하고 있어요..."):
                        feedback = evaluate_evidence(
                            work,
                            character,
                            st.session_state.history[picked - 1],
                            evidence.strip(),
                        )
                    st.info(feedback)
                except Exception:
                    st.info("오답 번호는 맞았습니다. 작품 근거는 수업에서 친구들과 함께 다시 확인해 보세요.")

                png = make_result_png(
                    work,
                    character,
                    st.session_state.history,
                    picked,
                    evidence.strip(),
                )
                st.download_button(
                    "결과를 PNG로 저장",
                    data=png,
                    file_name=f"{work.get('title','작품')}_인물인터뷰_결과.png",
                    mime="image/png",
                    use_container_width=True,
                )
            else:
                st.warning("다시 생각해 보세요. 작품 속 사건의 순서, 인물의 관계, 행동의 이유를 확인해 보세요.")

        if st.button("🔄 다시 인터뷰하기", use_container_width=True):
            reset_interview()
            st.rerun()
