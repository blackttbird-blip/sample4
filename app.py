import io
import json
import os
import random
import re
import uuid
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
from docx import Document


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="소설 속 인물 인터뷰",
    page_icon="📚",
    layout="centered",
)

MODEL = "gemini-2.5-flash"
MAX_TURNS = 5

DATA_DIR = Path("data")
WORKS_FILE = DATA_DIR / "works.json"

DATA_DIR.mkdir(exist_ok=True)


# =========================================================
# Secrets
# =========================================================

def secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def api_key():
    return secret("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")


# =========================================================
# Gemini 연결
# =========================================================

@st.cache_resource
def client():

    if not api_key():
        raise RuntimeError("GEMINI_API_KEY가 없습니다.")

    return genai.Client(
        api_key=api_key()
    )


def ask_gemini(
    prompt,
    search=False,
    temperature=0.3,
):

    """
    search=True이면 Google Search를 먼저 사용한다.

    검색 기능을 사용할 수 없는 경우에는
    Gemini 자체 지식으로 다시 한 번 답변한다.

    반환값
    1. 답변
    2. 실제 검색 사용 여부
    3. 검색 오류 메시지
    """

    if search:

        try:

            config = types.GenerateContentConfig(
                temperature=temperature,
                tools=[
                    types.Tool(
                        google_search=types.GoogleSearch()
                    )
                ],
            )

            response = client().models.generate_content(
                model=MODEL,
                contents=prompt,
                config=config,
            )

            return (
                (response.text or "").strip(),
                True,
                None,
            )

        except Exception as e:

            search_error = str(e)

            config = types.GenerateContentConfig(
                temperature=temperature
            )

            response = client().models.generate_content(
                model=MODEL,
                contents=prompt,
                config=config,
            )

            return (
                (response.text or "").strip(),
                False,
                search_error,
            )

    config = types.GenerateContentConfig(
        temperature=temperature
    )

    response = client().models.generate_content(
        model=MODEL,
        contents=prompt,
        config=config,
    )

    return (
        (response.text or "").strip(),
        False,
        None,
    )


# =========================================================
# 작품 저장
# =========================================================

def load_works():

    if not WORKS_FILE.exists():
        return []

    try:

        data = json.loads(
            WORKS_FILE.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(data, list):
            return data

        return []

    except Exception:

        return []


def save_works(works):

    WORKS_FILE.write_text(
        json.dumps(
            works,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def get_work(work_id):

    return next(
        (
            work
            for work in load_works()
            if work.get("id") == work_id
        ),
        None,
    )


# =========================================================
# 업로드 파일 읽기
# =========================================================

def read_hwpx(data):

    result = []

    with zipfile.ZipFile(
        io.BytesIO(data)
    ) as zip_file:

        names = sorted(
            name
            for name in zip_file.namelist()
            if name.lower().startswith(
                "contents/section"
            )
            and name.endswith(".xml")
        )

        for name in names:

            root = ET.fromstring(
                zip_file.read(name)
            )

            for element in root.iter():

                if (
                    element.tag.split("}")[-1]
                    == "t"
                    and element.text
                ):

                    result.append(
                        element.text
                    )

    return "\n".join(result)


def read_upload(file):

    extension = Path(
        file.name
    ).suffix.lower()

    data = file.getvalue()

    # TXT
    if extension == ".txt":

        for encoding in (
            "utf-8",
            "utf-8-sig",
            "cp949",
            "euc-kr",
        ):

            try:

                return data.decode(
                    encoding
                )

            except UnicodeDecodeError:

                pass

        return data.decode(
            "utf-8",
            errors="ignore",
        )

    # PDF
    if extension == ".pdf":

        reader = PdfReader(
            io.BytesIO(data)
        )

        return "\n".join(
            (
                page.extract_text()
                or ""
            )
            for page in reader.pages
        )

    # DOCX
    if extension == ".docx":

        document = Document(
            io.BytesIO(data)
        )

        return "\n".join(
            paragraph.text
            for paragraph
            in document.paragraphs
        )

    # HWPX
    if extension == ".hwpx":

        return read_hwpx(data)

    raise ValueError(
        "지원하지 않는 파일 형식입니다."
    )


def uploads_to_sources(files):

    sources = []
    errors = []

    for file in files or []:

        try:

            text = read_upload(
                file
            )

            text = re.sub(
                r"\n{3,}",
                "\n\n",
                text,
            ).strip()

            if text:

                sources.append(
                    {
                        "name": file.name,
                        "text": text,
                    }
                )

            else:

                errors.append(
                    f"{file.name}: "
                    "읽을 수 있는 텍스트가 없습니다."
                )

        except Exception as e:

            errors.append(
                f"{file.name}: {e}"
            )

    return sources, errors


def merge_sources(
    old_sources,
    new_sources,
):

    by_name = {
        item["name"]: item
        for item in old_sources
    }

    for item in new_sources:

        by_name[
            item["name"]
        ] = item

    return list(
        by_name.values()
    )


# =========================================================
# 질문과 관련 있는 자료 찾기
# =========================================================

def terms(text):

    return set(
        re.findall(
            r"[가-힣A-Za-z0-9]{2,}",
            text or "",
        )
    )


def context_for(
    work,
    question,
    character="",
):

    blocks = []

    source_types = (
        (
            "소설 본문",
            "novel_sources",
        ),
        (
            "학습자료",
            "learning_sources",
        ),
    )

    for label, key in source_types:

        for source in work.get(
            key,
            [],
        ):

            text = source.get(
                "text",
                "",
            )

            # 긴 텍스트를 나누기
            for i in range(
                0,
                len(text),
                1800,
            ):

                blocks.append(
                    (
                        label,
                        source.get(
                            "name",
                            "자료",
                        ),
                        text[
                            i:i + 2000
                        ],
                    )
                )

    if not blocks:
        return ""

    question_terms = terms(
        f"""
        {question}
        {character}
        {work.get("title", "")}
        {work.get("author", "")}
        """
    )

    scored = []

    for (
        label,
        name,
        chunk,
    ) in blocks:

        score = len(
            question_terms
            & terms(chunk)
        )

        if (
            character
            and character in chunk
        ):

            score += 8

        scored.append(
            (
                score,
                f"""
[{label} - {name}]
{chunk}
""",
            )
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    chosen = []
    total_length = 0

    for _, block in scored:

        if (
            total_length
            + len(block)
            > 50000
        ):

            continue

        chosen.append(
            block
        )

        total_length += len(
            block
        )

        if len(chosen) >= 18:
            break

    return "\n\n".join(
        chosen
    )


def has_teacher_material(
    work
):

    return bool(
        work.get(
            "novel_sources"
        )
        or work.get(
            "learning_sources"
        )
    )


# =========================================================
# 등장인물 자동 찾기
# =========================================================

def parse_chars(text):

    cleaned = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.I,
    )

    cleaned = cleaned.replace(
        "```",
        "",
    ).strip()

    try:

        data = json.loads(
            cleaned
        )

        if (
            isinstance(data, dict)
            and isinstance(
                data.get(
                    "characters"
                ),
                list,
            )
        ):

            return [
                str(item).strip()
                for item
                in data[
                    "characters"
                ]
                if str(
                    item
                ).strip()
            ][:10]

    except Exception:

        pass

    return [
        item.strip()
        for item in re.split(
            r"[,/\n]+",
            cleaned,
        )
        if (
            item.strip()
            and len(
                item.strip()
            )
            <= 20
        )
    ][:10]


def discover_characters(
    work
):

    context = context_for(
        work,
        "주요 등장인물",
    )

    prompt = f"""
너는 중학교 국어 수업용
문학 작품 분석 도우미다.

작품명:
{work.get("title", "")}

작가:
{work.get("author", "")}

교사가 업로드한 자료가 있다면
그 자료를 가장 우선해서 사용하라.

교사 자료가 없거나 부족하다면
반드시 작품명과 작가명을 기준으로
Google Search를 활용하여
정확한 작품을 확인하라.

동명의 다른 작품,
다른 작가의 작품,
영화나 드라마 각색판의 내용을
섞지 마라.

확신할 수 없는 인물 이름은
만들어내지 마라.

학생이 인터뷰하기 적절한
실제 등장인물 2~8명을 골라라.

교사 제공 자료:

{context if context else "(자료 없음)"}

반드시 다음 JSON 형식만 출력하라.

{{
    "characters": [
        "인물1",
        "인물2"
    ]
}}
"""

    text, used_search, error = (
        ask_gemini(
            prompt,
            search=not has_teacher_material(
                work
            ),
            temperature=0.1,
        )
    )

    return (
        parse_chars(text),
        used_search,
        error,
    )


# =========================================================
# 이전 대화 정리
# =========================================================

def history_text(
    history
):

    if not history:

        return (
            "(이전 대화 없음)"
        )

    return "\n".join(
        f"""
{i}. 학생 질문:
{item["question"]}

{item["character"]}의 답변:
{item["answer"]}
"""
        for i, item
        in enumerate(
            history,
            1,
        )
    )


# =========================================================
# 인물 답변 만들기
# =========================================================

def make_answer(
    work,
    character,
    question,
    history,
    mistake=False,
):

    context = context_for(
        work,
        question,
        character,
    )

    if mistake:

        mode = """
이번 차례는
'의도된 오답' 차례다.

학생에게
오답이라는 사실을
절대 밝히지 마라.

질문과 직접 관련된
핵심 사실 가운데
딱 하나만
미묘하게 틀리게 답하라.

적절한 오류 예시는 다음과 같다.

- 사건의 원인과 결과 뒤집기
- 사건의 순서 바꾸기
- 인물의 동기 왜곡
- 인물 관계 왜곡
- 본문과 충돌하는
  그럴듯한 세부 정보

말도 안 되는 거짓말은
만들지 마라.

학생이 작품을 다시 읽으면
구체적인 근거를 찾아
반박할 수 있어야 한다.
"""

    else:

        mode = """
이번 차례는
정확한 답변 차례다.

의도적으로
틀린 정보를 넣지 마라.

확실하지 않은 내용을
사실처럼 만들어내지 마라.
"""

    prompt = f"""
너는 중학교 국어 수업의
'소설 속 인물 인터뷰' 활동에서
등장인물 역할을 맡는다.

작품명:
{work.get("title", "")}

작가:
{work.get("author", "")}

현재 역할:
{character}

자료 사용 우선순위는 다음과 같다.

1. 교사가 업로드한 소설 본문
2. 교사가 업로드한 학습자료
3. 작품 속 인물의 행동·관계·상황
4. 합리적인 문학적 추론
5. 위 자료가 없거나 부족한 경우에만
   작품명과 작가명을 이용한
   Google Search 또는 모델 지식

교사가 제공한 자료와
외부 정보가 충돌하면
반드시 교사 자료를 따른다.

{mode}

답변 규칙:

- 반드시 {character} 본인이
  말하는 1인칭 말투로 답한다.

- 해설자처럼 설명하지 않는다.

- 학생 질문에 직접 답한다.

- 중학생이 이해하기 쉬운
  2~5문장 정도로 답한다.

- 질문이 지나치게 짧거나
  의미가 불분명해도
  질문 한 번으로 처리한다.

- 이 경우 인물의 입장에서
  짧게 답한 뒤
  작품과 관련된 질문을
  자연스럽게 유도한다.

- 긴 원문 인용은 하지 않는다.

- AI, 프롬프트,
  오답 차례, 검증 등
  내부 정보를 말하지 않는다.

이전 대화:

{history_text(history)}

이번 학생 질문:

{question}

교사가 제공한 자료 중
관련성이 높은 부분:

{context if context else "(교사 제공 자료 없음)"}

이제 {character}의 입장에서
자연스럽게 답하라.
"""

    return ask_gemini(
        prompt,
        search=not has_teacher_material(
            work
        ),
        temperature=(
            0.55
            if mistake
            else 0.3
        ),
    )


# =========================================================
# 의도된 오답 검증
# =========================================================

def validate_mistake(
    work,
    character,
    question,
    answer,
):

    context = context_for(
        work,
        question,
        character,
    )

    prompt = f"""
너는 중학교 국어 수업용
오답 검증기다.

작품명:
{work.get("title", "")}

작가:
{work.get("author", "")}

인물:
{character}

학생 질문:
{question}

후보 답변:
{answer}

관련 교사 자료:
{context if context else "(자료 없음)"}

다음 조건을 모두 확인하라.

1.
질문과 직접 관련된 답변인가?

2.
핵심 사실 가운데 하나가
작품 내용과 충돌하는가?

3.
학생이 작품을 다시 읽고
구체적인 근거를 찾아
반박할 수 있는가?

4.
너무 황당하거나
눈에 띄는 거짓말은 아닌가?

5.
오답이라는 사실을
스스로 밝히지 않았는가?

반드시 JSON만 출력하라.

{{
    "valid": true
}}
"""

    text, _, _ = ask_gemini(
        prompt,
        search=not has_teacher_material(
            work
        ),
        temperature=0.1,
    )

    cleaned = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.I,
    )

    cleaned = cleaned.replace(
        "```",
        "",
    ).strip()

    try:

        result = json.loads(
            cleaned
        )

        return bool(
            result.get(
                "valid",
                False,
            )
        )

    except Exception:

        return False


def generate_answer(
    work,
    character,
    question,
    history,
    mistake,
):

    if not mistake:

        return make_answer(
            work,
            character,
            question,
            history,
            False,
        )

    last_result = None

    # 오답 검증에 실패하면
    # 최대 3번까지 다시 생성
    for _ in range(3):

        last_result = make_answer(
            work,
            character,
            question,
            history,
            True,
        )

        try:

            if validate_mistake(
                work,
                character,
                question,
                last_result[0],
            ):

                return last_result

        except Exception:

            return last_result

    return last_result


# =========================================================
# 학생이 찾은 근거 평가
# =========================================================

def evaluate_evidence(
    work,
    character,
    item,
    evidence,
):

    context = context_for(
        work,
        item["question"],
        character,
    )

    prompt = f"""
너는 중학교 국어 교사 보조자다.

작품명:
{work.get("title", "")}

작가:
{work.get("author", "")}

학생 질문:
{item["question"]}

AI 답변:
{item["answer"]}

학생이 제시한 작품 근거:
{evidence}

관련 교사 자료:
{context if context else "(자료 없음)"}

학생이 제시한 근거가
오답을 반박하는 데
적절한지 평가하라.

적절하다면
왜 좋은 근거인지
2~3문장으로 설명하라.

근거가 부족하다면
정답을 바로 알려주지 말고
작품의 어느 부분을
다시 살펴봐야 하는지
힌트를 줘라.

작품에 없는 내용을
만들어내지 마라.
"""

    text, _, _ = ask_gemini(
        prompt,
        search=not has_teacher_material(
            work
        ),
        temperature=0.2,
    )

    return text


# =========================================================
# 세션 상태
# =========================================================

def init_state():

    defaults = {

        "admin_ok":
            False,

        "started":
            False,

        "work_id":
            None,

        "character":
            None,

        "history":
            [],

        "mistake_turn":
            None,

        "finished":
            False,

        "notice":
            None,
    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[
                key
            ] = value


def reset_interview():

    values = {

        "started":
            False,

        "work_id":
            None,

        "character":
            None,

        "history":
            [],

        "mistake_turn":
            None,

        "finished":
            False,

        "notice":
            None,
    }

    for key, value in values.items():

        st.session_state[
            key
        ] = value


init_state()


# =========================================================
# 화면 스타일
# =========================================================

st.markdown(
    """
<style>

.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

.progress {
    padding: 0.65rem;
    border: 1px solid #ddd;
    border-radius: 12px;
    text-align: center;
    font-weight: 700;
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# 메뉴
# =========================================================

page = st.sidebar.radio(
    "메뉴",
    [
        "학생 화면",
        "교사 설정",
    ],
)


# =========================================================
# 교사 설정 화면
# =========================================================

if page == "교사 설정":

    st.title(
        "⚙️ 교사 설정"
    )

    # 관리자 로그인
    if not st.session_state.admin_ok:

        password = st.text_input(
            "관리자 비밀번호",
            type="password",
        )

        if st.button(
            "로그인",
            use_container_width=True,
        ):

            if password == str(
                secret(
                    "ADMIN_PASSWORD",
                    "6460",
                )
            ):

                st.session_state.admin_ok = True

                st.rerun()

            else:

                st.error(
                    "비밀번호가 맞지 않습니다."
                )

        st.stop()

    st.caption(
        "작품명과 작가 이름만 필수입니다. "
        "나머지는 공란이어도 됩니다."
    )

    works = load_works()

    labels = [
        "➕ 새 작품 만들기"
    ]

    labels += [
        f"{work['title']} — {work['author']}"
        for work in works
    ]

    choice = st.selectbox(
        "작품 선택",
        labels,
    )

    index = (
        labels.index(choice)
        - 1
    )

    if index >= 0:

        current = works[
            index
        ]

    else:

        current = None

    # -----------------------------------------------------
    # 작품명 / 작가
    # -----------------------------------------------------

    title = st.text_input(
        "작품명 *",
        value=(
            current.get(
                "title",
                "",
            )
            if current
            else ""
        ),
    )

    author = st.text_input(
        "작가 이름 *",
        value=(
            current.get(
                "author",
                "",
            )
            if current
            else ""
        ),
    )

    # -----------------------------------------------------
    # 등장인물
    # -----------------------------------------------------

    characters_text = st.text_input(
        "등장인물 (선택)",
        value=(
            ", ".join(
                current.get(
                    "characters",
                    [],
                )
            )
            if current
            else ""
        ),
        placeholder=(
            "비워 두면 "
            "Gemini가 자동으로 찾습니다."
        ),
    )

    # -----------------------------------------------------
    # 소설 파일
    # -----------------------------------------------------

    st.subheader(
        "소설 전문/본문 파일 (선택)"
    )

    if (
        current
        and current.get(
            "novel_sources"
        )
    ):

        st.caption(
            "현재 저장된 파일: "
            + ", ".join(
                item["name"]
                for item
                in current[
                    "novel_sources"
                ]
            )
        )

    novel_files = (
        st.file_uploader(
            "TXT, PDF, DOCX, HWPX",
            type=[
                "txt",
                "pdf",
                "docx",
                "hwpx",
            ],
            accept_multiple_files=True,
            key=(
                "novel_"
                + (
                    current.get(
                        "id"
                    )
                    if current
                    else "new"
                )
            ),
        )
    )

    if current:

        clear_novel = (
            st.checkbox(
                "기존 소설 파일 삭제",
                value=False,
            )
        )

    else:

        clear_novel = False

    # -----------------------------------------------------
    # 학습자료 파일
    # -----------------------------------------------------

    st.subheader(
        "학습자료 파일 (선택)"
    )

    if (
        current
        and current.get(
            "learning_sources"
        )
    ):

        st.caption(
            "현재 저장된 파일: "
            + ", ".join(
                item["name"]
                for item
                in current[
                    "learning_sources"
                ]
            )
        )

    learning_files = (
        st.file_uploader(
            "TXT, PDF, DOCX, HWPX",
            type=[
                "txt",
                "pdf",
                "docx",
                "hwpx",
            ],
            accept_multiple_files=True,
            key=(
                "learning_"
                + (
                    current.get(
                        "id"
                    )
                    if current
                    else "new"
                )
            ),
        )
    )

    if current:

        clear_learning = (
            st.checkbox(
                "기존 학습자료 삭제",
                value=False,
            )
        )

    else:

        clear_learning = False

    st.info(
        "파일을 올리지 않아도 됩니다. "
        "그 경우 작품명과 작가 이름을 기준으로 "
        "Google 검색을 활용합니다."
    )

    # -----------------------------------------------------
    # 저장 / 삭제
    # -----------------------------------------------------

    column1, column2 = st.columns(
        [
            3,
            1,
        ]
    )

    save_button = (
        column1.button(
            "💾 저장",
            type="primary",
            use_container_width=True,
        )
    )

    delete_button = (
        column2.button(
            "🗑️ 삭제",
            use_container_width=True,
            disabled=(
                current is None
            ),
        )
    )

    # 삭제
    if (
        delete_button
        and current
    ):

        save_works(
            [
                work
                for work
                in works
                if (
                    work["id"]
                    != current["id"]
                )
            ]
        )

        st.rerun()

    # 저장
    if save_button:

        if (
            not title.strip()
            or not author.strip()
        ):

            st.error(
                "작품명과 작가 이름은 "
                "반드시 입력해 주세요."
            )

            st.stop()

        # 새 파일 읽기
        (
            new_novel_sources,
            novel_errors,
        ) = uploads_to_sources(
            novel_files
        )

        (
            new_learning_sources,
            learning_errors,
        ) = uploads_to_sources(
            learning_files
        )

        for error in (
            novel_errors
            + learning_errors
        ):

            st.warning(
                error
            )

        # 기존 자료
        if (
            current
            and not clear_novel
        ):

            old_novel_sources = (
                current.get(
                    "novel_sources",
                    [],
                )
            )

        else:

            old_novel_sources = []

        if (
            current
            and not clear_learning
        ):

            old_learning_sources = (
                current.get(
                    "learning_sources",
                    [],
                )
            )

        else:

            old_learning_sources = []

        # 등장인물
        manual_characters = [
            item.strip()
            for item
            in re.split(
                r"[,/\n]+",
                characters_text,
            )
            if item.strip()
        ]

        work = {

            "id":
                (
                    current["id"]
                    if current
                    else str(
                        uuid.uuid4()
                    )
                ),

            "title":
                title.strip(),

            "author":
                author.strip(),

            "characters":
                manual_characters,

            "novel_sources":
                merge_sources(
                    old_novel_sources,
                    new_novel_sources,
                ),

            "learning_sources":
                merge_sources(
                    old_learning_sources,
                    new_learning_sources,
                ),
        }

        # 등장인물이 비어 있으면 자동 검색
        if (
            not manual_characters
            and api_key()
        ):

            try:

                with st.spinner(
                    "등장인물을 자동으로 "
                    "찾고 있어요..."
                ):

                    (
                        automatic_characters,
                        _,
                        _,
                    ) = discover_characters(
                        work
                    )

                if automatic_characters:

                    work[
                        "characters"
                    ] = automatic_characters

            except Exception:

                pass

        # 기존 작품 수정
        if current:

            current_id = current[
                "id"
            ]

            works = [
                (
                    work
                    if item["id"]
                    == current_id
                    else item
                )
                for item
                in works
            ]

        # 새 작품
        else:

            works.append(
                work
            )

        save_works(
            works
        )

        st.success(
            "저장했습니다."
        )

        st.rerun()

    st.caption(
        "※ 현재 프로토타입은 "
        "Streamlit 서버의 로컬 JSON에 "
        "저장됩니다. "
        "앱을 재배포하면 "
        "데이터가 사라질 수 있습니다."
    )


# =========================================================
# 학생 화면
# =========================================================

else:

    st.title(
        "📚 소설 속 인물 인터뷰"
    )

    st.caption(
        "등장인물에게 5번 질문하고, "
        "답변 속 한 번의 오류를 "
        "작품 근거로 찾아보세요."
    )

    # API 키 확인
    if not api_key():

        st.error(
            "GEMINI_API_KEY가 없습니다. "
            "Streamlit Cloud의 Secrets에 "
            "Gemini API 키를 넣어 주세요."
        )

        st.stop()

    works = load_works()

    if not works:

        st.info(
            "교사 설정에서 "
            "작품을 먼저 등록해 주세요."
        )

        st.stop()

    # =====================================================
    # 인터뷰 시작 전
    # =====================================================

    if not st.session_state.started:

        column1, column2 = (
            st.columns(2)
        )

        labels = [
            f"{work['title']} — {work['author']}"
            for work
            in works
        ]

        selected = (
            column1.selectbox(
                "작품",
                labels,
            )
        )

        work = works[
            labels.index(
                selected
            )
        ]

        characters = (
            work.get(
                "characters",
                [],
            )
        )

        # 등장인물이 없으면 자동 검색
        if not characters:

            try:

                with st.spinner(
                    "등장인물을 찾고 있어요..."
                ):

                    (
                        characters,
                        used_search,
                        search_error,
                    ) = discover_characters(
                        work
                    )

                if characters:

                    work[
                        "characters"
                    ] = characters

                    work_id = work[
                        "id"
                    ]

                    works = [
                        (
                            work
                            if item["id"]
                            == work_id
                            else item
                        )
                        for item
                        in works
                    ]

                    save_works(
                        works
                    )

                if (
                    search_error
                    and not used_search
                ):

                    st.caption(
                        "검색 연결을 사용할 수 없어 "
                        "Gemini 자체 지식으로 "
                        "등장인물을 찾았습니다."
                    )

            except Exception:

                characters = []

        # 등장인물 선택
        if characters:

            character = (
                column2.selectbox(
                    "인터뷰할 인물",
                    characters,
                )
            )

        else:

            character = (
                column2.text_input(
                    "인터뷰할 인물 이름"
                )
            )

        # 인터뷰 시작
        if st.button(
            "인터뷰 시작",
            type="primary",
            use_container_width=True,
        ):

            if not str(
                character
            ).strip():

                st.warning(
                    "인터뷰할 인물을 "
                    "선택해 주세요."
                )

            else:

                st.session_state.started = True

                st.session_state.work_id = (
                    work["id"]
                )

                st.session_state.character = (
                    str(
                        character
                    ).strip()
                )

                st.session_state.history = []

                # 1~5 중 한 번을
                # 오답 차례로 무작위 선택
                st.session_state.mistake_turn = (
                    random.randint(
                        1,
                        MAX_TURNS,
                    )
                )

                st.session_state.finished = False

                st.session_state.notice = None

                st.rerun()

        st.stop()

    # =====================================================
    # 인터뷰 진행
    # =====================================================

    work = get_work(
        st.session_state.work_id
    )

    if not work:

        st.error(
            "작품 정보를 "
            "찾을 수 없습니다."
        )

        st.stop()

    character = (
        st.session_state.character
    )

    count = len(
        st.session_state.history
    )

    left, right = st.columns(
        [
            3,
            1,
        ]
    )

    left.subheader(
        f"{work['title']} · {character}"
    )

    left.caption(
        f"{work['author']} 작품"
    )

    right.markdown(
        f"""
<div class="progress">
질문 {min(count + 1, 5)} / 5
</div>
""",
        unsafe_allow_html=True,
    )

    # 진행 표시
    dots = " ".join(
        (
            "●"
            if i <= count
            else "○"
        )
        for i in range(
            1,
            6,
        )
    )

    st.markdown(
        "**진행:** "
        + dots
    )

    # 이전 질문/답변 출력
    for (
        index,
        item,
    ) in enumerate(
        st.session_state.history,
        1,
    ):

        with st.chat_message(
            "user"
        ):

            st.markdown(
                f"""
**Q{index}.**
{item["question"]}
"""
            )

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                item["answer"]
            )

    # =====================================================
    # 질문 입력
    # =====================================================

    if (
        count < MAX_TURNS
        and not st.session_state.finished
    ):

        with st.form(
            "question_form",
            clear_on_submit=True,
        ):

            question = (
                st.text_input(
                    "질문",
                    max_chars=300,
                    placeholder=(
                        f"{character}에게 "
                        "한 가지 질문을 해 보세요."
                    ),
                )
            )

            submit = (
                st.form_submit_button(
                    "질문하기",
                    type="primary",
                    use_container_width=True,
                )
            )

        if submit:

            if not question.strip():

                st.warning(
                    "질문을 입력해 주세요."
                )

            else:

                turn = count + 1

                is_mistake = (
                    turn
                    == st.session_state.mistake_turn
                )

                try:

                    with st.spinner(
                        f"{character}가 "
                        "답변을 생각하고 있어요..."
                    ):

                        (
                            answer,
                            used_search,
                            search_error,
                        ) = generate_answer(
                            work,
                            character,
                            question.strip(),
                            st.session_state.history,
                            is_mistake,
                        )

                    st.session_state.history.append(
                        {
                            "question":
                                question.strip(),

                            "answer":
                                answer,

                            "character":
                                character,
                        }
                    )

                    if (
                        search_error
                        and not used_search
                    ):

                        st.session_state.notice = (
                            "웹 검색을 사용할 수 없어 "
                            "Gemini 자체 지식으로 "
                            "답한 항목이 있습니다."
                        )

                    if (
                        len(
                            st.session_state.history
                        )
                        == MAX_TURNS
                    ):

                        st.session_state.finished = True

                    st.rerun()

                except Exception as e:

                    st.error(
                        "답변 생성에 실패했습니다: "
                        + str(e)
                    )

    # 검색 실패 안내
    if st.session_state.notice:

        st.caption(
            "※ "
            + st.session_state.notice
        )

    # =====================================================
    # 5회 질문 완료
    # =====================================================

    if st.session_state.finished:

        st.divider()

        st.subheader(
            "🔎 AI의 오류를 찾아보세요"
        )

        st.write(
            "5개의 답변 중 "
            "작품 내용과 어긋나는 답변은 "
            "**딱 하나**입니다."
        )

        with st.form(
            "result_form"
        ):

            picked = st.selectbox(
                "틀린 답변 번호",
                [
                    1,
                    2,
                    3,
                    4,
                    5,
                ],
            )

            evidence = st.text_area(
                "작품 속 근거",
                height=130,
                placeholder=(
                    "왜 그 답변이 틀렸는지 "
                    "사건, 행동, 대사 등을 "
                    "근거로 적어 보세요."
                ),
            )

            check = (
                st.form_submit_button(
                    "확인하기",
                    type="primary",
                    use_container_width=True,
                )
            )

        if check:

            # 근거 미입력
            if not evidence.strip():

                st.warning(
                    "작품 속 근거를 "
                    "적어 주세요."
                )

            # 오답 번호 맞음
            elif (
                picked
                == st.session_state.mistake_turn
            ):

                st.success(
                    "오답의 위치를 "
                    "정확히 찾았습니다!"
                )

                try:

                    with st.spinner(
                        "근거를 확인하고 있어요..."
                    ):

                        feedback = (
                            evaluate_evidence(
                                work,
                                character,
                                st.session_state.history[
                                    picked - 1
                                ],
                                evidence.strip(),
                            )
                        )

                    st.info(
                        feedback
                    )

                except Exception:

                    st.info(
                        "오답 번호는 맞았습니다. "
                        "작품 근거는 수업에서 "
                        "함께 확인해 보세요."
                    )

            # 오답 번호 틀림
            else:

                st.warning(
                    "다시 생각해 보세요. "
                    "사건의 순서, 인물 관계, "
                    "행동의 이유를 "
                    "확인해 보세요."
                )

        # 다시 시작
        if st.button(
            "🔄 다시 인터뷰하기",
            use_container_width=True,
        ):

            reset_interview()

            st.rerun()
