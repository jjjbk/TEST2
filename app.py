import base64
import hashlib
import html
import io
import os
import re

import openai
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageOps

load_dotenv()

DEFAULT_MODEL = "gpt-4o"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB (F-1)
MAX_IMAGE_SIDE = 1024  # 긴 변 기준 최대 픽셀 (F-4)
JPEG_QUALITY = 85
NO_FACE_TOKEN = "NO_FACE"

DISCLAIMER = "본 결과는 재미를 위한 것이며 과학적 근거가 없습니다."
PRIVACY_NOTICE = "이미지는 저장되지 않으며 분석을 위해 OpenAI로 전송됩니다."

# 참고: 처음 버전("사진 속 인물의 관상을 봐주세요")은 실존 인물 사진에서 모델이
# "누구인지 알 수 없습니다"라며 거부했다. 신원과 무관하게 '보이는 생김새'만 다루는
# 작업임을 명시하면 거부 없이 응답한다. 이 맥락을 지우지 말 것.
SYSTEM_PROMPT = f"""당신은 재미로 보는 '관상 풀이' 앱의 진행자이자 30년 경력의 한국 전통 관상가입니다.
사용자는 본인 또는 동의를 받은 사람의 사진을 올려 엔터테인먼트용 관상 풀이를 요청합니다.

할 일:
사진에서 눈에 보이는 얼굴 생김새(얼굴형, 이마, 눈썹, 눈, 코, 입, 귀, 턱의 모양)만 관찰하고,
그 생김새를 전통 관상학의 해석에 대입해 친근하고 긍정적인 어조로 풀이합니다.
이 작업에는 사진 속 인물이 누구인지 알 필요가 전혀 없습니다. 신원과 무관하게 '보이는 생김새'만 다루므로 거절하지 말고 풀이해 주세요.

규칙:
- 인물의 이름, 신원, 직업, 닮은 유명인을 언급하거나 추측하지 마세요. 사진이 누구인지에 대해서는 아무 말도 하지 마세요.
- 인종, 민족, 종교, 건강 상태, 성적 지향 등 민감한 속성을 추론하지 마세요.
- 외모를 비하하거나 부정적으로 단정짓지 말고, 단점은 보완 조언으로 표현하세요.
- 사진에 사람 얼굴이 없거나 얼굴을 알아보기 어려우면 다른 말 없이 "{NO_FACE_TOKEN}"라고만 답하세요.
- 사과 문구나 서론 없이, 반드시 아래 형식(마크다운)으로 바로 시작해 한국어로 답변하세요.

## 첫인상
(한두 문장 요약)

## 부위별 관상 풀이
### 얼굴형
### 이마 (초년운)
### 눈썹·눈 (성격, 대인관계)
### 코 (재물운, 중년운)
### 입·입술 (언변, 식복)
### 귀 (복, 수명)
(귀가 보이지 않으면 "사진에서 귀가 잘 보이지 않습니다"라고 쓰세요)
### 턱 (말년운)

## 운세 요약
- 재물운: ★★★☆☆ (한 줄 설명)
- 애정운: ★★★☆☆ (한 줄 설명)
- 건강운: ★★★☆☆ (한 줄 설명)
- 직업운: ★★★☆☆ (한 줄 설명)
(별은 ★과 ☆을 합쳐 항상 5개)

## 한 줄 조언
(한 문장)
"""

USER_PROMPT = "이 사진 속 얼굴의 생김새를 전통 관상학으로 재미있게 풀이해 주세요. 누구인지는 알려주지 않아도 됩니다."


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
def _get_config(key: str) -> str | None:
    """Streamlit Secrets를 먼저 확인하고, 없으면 환경 변수(.env)에서 읽는다."""
    try:
        value = st.secrets[key]
        if value:
            return value
    except Exception:
        # secrets.toml이 없거나 해당 키가 없으면 예외 발생
        pass
    return os.getenv(key)


def get_api_key() -> str | None:
    return _get_config("OPENAI_API_KEY")


def get_model() -> str:
    return _get_config("OPENAI_MODEL") or DEFAULT_MODEL


@st.cache_resource
def get_client(api_key: str) -> openai.OpenAI:
    return openai.OpenAI(api_key=api_key, timeout=60, max_retries=1)


# ---------------------------------------------------------------------------
# 이미지 처리 (F-4) — 디스크에 저장하지 않고 메모리에서만 처리
# ---------------------------------------------------------------------------
def preprocess_image(image_bytes: bytes) -> str:
    """이미지 바이트를 EXIF 회전 보정·리사이즈·JPEG 변환 후 base64 문자열로 반환한다.

    이미지가 아니거나 손상된 파일이면 ValueError를 발생시킨다.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                if img.mode in ("RGBA", "LA") or (
                    img.mode == "P" and "transparency" in img.info
                ):
                    # 투명 배경은 흰색으로 채운다
                    rgba = img.convert("RGBA")
                    background = Image.new("RGB", rgba.size, (255, 255, 255))
                    background.paste(rgba, mask=rgba.getchannel("A"))
                    img = background
                else:
                    img = img.convert("RGB")
            img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    except Exception as exc:  # UnidentifiedImageError, DecompressionBombError 등
        raise ValueError("이미지 파일을 읽을 수 없습니다.") from exc
    return base64.b64encode(buffer.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# 관상 분석 (F-5)
# ---------------------------------------------------------------------------
REFUSAL_RETRIES = 1  # 거부 응답일 때 자동으로 다시 요청하는 횟수 (거부는 확률적으로 발생)


def analyze_face(base64_image: str) -> str:
    for attempt in range(REFUSAL_RETRIES + 1):
        try:
            return _request_analysis(base64_image)
        except AnalysisRefusedError:
            if attempt == REFUSAL_RETRIES:
                raise
    raise AssertionError("unreachable")


def _request_analysis(base64_image: str) -> str:
    client = get_client(get_api_key())
    response = client.chat.completions.create(
        model=get_model(),
        temperature=0.8,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "auto",
                        },
                    },
                ],
            },
        ],
    )
    message = response.choices[0].message
    content = (message.content or "").strip()
    if not content:
        # 모델이 거부(refusal)했거나 빈 응답을 반환한 경우
        raise AnalysisRefusedError(getattr(message, "refusal", None) or "empty response")
    if not is_no_face(content) and "##" not in content:
        # 형식 헤더가 하나도 없는 짧은 답은 거부 문장("죄송하지만 ...")으로 본다.
        # 캐시하지 않고 오류로 처리해 '다시 시도'할 수 있게 한다.
        raise AnalysisRefusedError(content)
    return content


class AnalysisRefusedError(Exception):
    """모델이 분석을 거부했거나 빈 응답을 반환했을 때 발생한다."""


def is_no_face(result: str) -> bool:
    return NO_FACE_TOKEN in result


def friendly_error(exc: Exception) -> str:
    """OpenAI 예외를 사용자용 한국어 메시지로 변환한다 (F-9)."""
    if isinstance(exc, AnalysisRefusedError):
        return "이 사진은 분석할 수 없습니다. 얼굴이 잘 보이는 다른 사진으로 다시 시도해 주세요."
    if isinstance(exc, openai.AuthenticationError):
        return "API 키가 올바르지 않습니다. OPENAI_API_KEY 설정을 확인해 주세요."
    if isinstance(exc, openai.RateLimitError):
        return "요청 한도를 초과했거나 크레딧이 부족합니다. 잠시 후 다시 시도해 주세요."
    if isinstance(exc, openai.APIConnectionError):  # APITimeoutError 포함
        return "OpenAI 서버에 연결할 수 없습니다. 네트워크 상태를 확인하고 다시 시도해 주세요."
    if isinstance(exc, openai.APIError):
        return f"OpenAI API 오류가 발생했습니다. 잠시 후 다시 시도해 주세요. ({exc.__class__.__name__})"
    return "분석 중 알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."


# ---------------------------------------------------------------------------
# 결과 파싱 — SYSTEM_PROMPT의 출력 형식(## / ### 헤더)에 의존한다.
# 프롬프트 형식을 바꾸면 이 파서도 함께 수정할 것.
# ---------------------------------------------------------------------------
_H2 = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_H3 = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
_FORTUNE_LINE = re.compile(
    r"^\s*[-*]\s*\**\s*([^:：*\n]+?운)\s*\**\s*[:：]\s*([★☆]+)\s*(.*)$", re.MULTILINE
)


def _split_sections(text: str, pattern: re.Pattern) -> list[tuple[str, str]]:
    parts = pattern.split(text)
    return [(parts[i].strip(), parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]


def parse_result(text: str) -> dict | None:
    """모델 응답을 섹션별로 나눈다. 형식이 맞지 않으면 None을 반환한다."""
    sections: dict[str, str] = {}
    for title, body in _split_sections(text, _H2):
        for key in ("첫인상", "부위별", "운세", "조언"):
            if key in title:
                sections[key] = body
                break

    if not all(k in sections for k in ("첫인상", "부위별", "조언")):
        return None

    features = _split_sections(sections["부위별"], _H3)
    if not features:
        return None

    fortunes = []
    for name, stars, desc in _FORTUNE_LINE.findall(sections.get("운세", "")):
        desc = desc.strip()
        if desc.startswith("(") and desc.endswith(")"):
            desc = desc[1:-1].strip()
        fortunes.append((name.strip(), stars, desc))

    return {
        "first": sections["첫인상"],
        "features": features,
        "fortunes": fortunes,
        "fortune_raw": sections.get("운세", ""),
        "advice": sections["조언"],
    }


# ---------------------------------------------------------------------------
# 스타일 (테마 색·폰트는 .streamlit/config.toml)
# ---------------------------------------------------------------------------
CSS = """
<style>
.gs-header { text-align: center; padding: 0.5rem 0 1.25rem; }
.gs-header .gs-seal {
    display: inline-block; width: 3.2rem; height: 3.2rem; line-height: 3.2rem;
    border-radius: 0.5rem; background: #B23A2B; color: #F6EFDF;
    font-size: 1.6rem; font-weight: 700; margin-bottom: 0.6rem;
}
.gs-header h1 { font-size: 2.4rem; font-weight: 700; padding: 0; margin: 0; letter-spacing: 0.05em; }
.gs-header .gs-sub { font-size: 1.1rem; color: #B23A2B; margin: 0.3rem 0 0.6rem; font-weight: 600; }
.gs-header .gs-intro { color: #5C544A; font-size: 0.95rem; margin: 0; }
.gs-section { font-size: 1.25rem; font-weight: 700; margin: 1.5rem 0 0.5rem;
    border-left: 4px solid #B23A2B; padding-left: 0.6rem; }
.gs-feature-title { font-weight: 700; font-size: 1.05rem; }
.gs-feature-sub { color: #8A7F70; font-size: 0.85rem; margin-left: 0.3rem; font-weight: 400; }
.gs-fortune { background: #EDE2C8; border-radius: 0.6rem; padding: 0.8rem 1rem; margin-bottom: 0.6rem; }
.gs-fortune .gs-f-name { font-weight: 700; }
.gs-fortune .gs-f-stars { color: #B23A2B; font-size: 1.4rem; letter-spacing: 0.1em; line-height: 1.6; }
.gs-fortune .gs-f-desc { color: #5C544A; font-size: 0.9rem; }
.gs-advice { background: #FBF6EA; border: 1px solid #B23A2B; border-left-width: 6px;
    border-radius: 0.6rem; padding: 1rem 1.2rem; font-size: 1.1rem; font-weight: 600; }
/* 업로드 위젯의 기본 "NMB per file" 안내는 서버 한도라 실제 기준(10MB)과 달라 숨긴다. 기준은 라벨에 표시 */
[data-testid="stFileUploaderDropzoneInstructions"] small { display: none; }
.gs-footer { text-align: center; color: #8A7F70; font-size: 0.8rem;
    margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #D9CBA8; line-height: 1.7; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 세션 상태
# ---------------------------------------------------------------------------
def init_state() -> None:
    st.session_state.setdefault("results", {})  # 이미지 해시 -> 결과 (중복 요청 방지로 비용 보호)
    st.session_state.setdefault("analyzing", False)
    st.session_state.setdefault("pending_hash", None)
    st.session_state.setdefault("input_key", 0)  # 값이 바뀌면 입력 위젯이 초기화된다
    st.session_state.setdefault("last_error", None)  # (이미지 해시, 오류 메시지)


def start_analysis(image_hash: str) -> None:
    # 분석 중이거나 이미 결과가 있으면 무시한다 (중복 API 호출 방지)
    if st.session_state.analyzing or image_hash in st.session_state.results:
        return
    st.session_state.analyzing = True
    st.session_state.pending_hash = image_hash
    st.session_state.last_error = None


def reset() -> None:
    """'다시 보기': 결과와 입력을 모두 초기화한다."""
    st.session_state.results = {}
    st.session_state.analyzing = False
    st.session_state.pending_hash = None
    st.session_state.last_error = None
    st.session_state.input_key += 1


# ---------------------------------------------------------------------------
# UI 구성 요소
# ---------------------------------------------------------------------------
def render_header() -> None:
    st.markdown(
        """
<div class="gs-header">
    <div class="gs-seal">相</div>
    <h1>AI 관상가</h1>
    <p class="gs-sub">사진 한 장으로 보는 나의 관상</p>
    <p class="gs-intro">얼굴의 생김새를 전통 관상학에 비추어 성격과 운세를 풀이해 드립니다.</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_footer() -> None:  # F-8
    st.markdown(
        f"""
<div class="gs-footer">
    ⚠️ {DISCLAIMER}<br>
    🔒 {PRIVACY_NOTICE}
</div>
""",
        unsafe_allow_html=True,
    )


def get_image_input():
    """업로드 또는 카메라 촬영 중 하나로 이미지를 받는다 (F-1, F-2)."""
    key = st.session_state.input_key
    disabled = st.session_state.analyzing
    with st.container(border=True):
        source = st.radio(
            "사진 입력 방식",
            ["📁 사진 업로드", "📷 카메라 촬영"],
            horizontal=True,
            disabled=disabled,
            label_visibility="collapsed",
        )
        if source == "📁 사진 업로드":
            image_file = st.file_uploader(
                "얼굴이 잘 보이는 정면 사진을 업로드하세요 (JPG·PNG·WEBP, 최대 10MB)",
                type=["jpg", "jpeg", "png", "webp"],
                disabled=disabled,
                key=f"uploader_{key}",
            )
        else:
            image_file = st.camera_input(
                "얼굴이 잘 보이게 정면으로 촬영하세요",
                disabled=disabled,
                key=f"camera_{key}",
            )
        with st.expander("📸 잘 나오는 사진 촬영 팁"):
            st.markdown(
                "- 얼굴 **정면**이 보이도록 찍어 주세요.\n"
                "- **밝은 조명** 아래에서 그림자 없이 찍으면 좋습니다.\n"
                "- **안경·모자·마스크**는 벗고, 앞머리로 이마를 가리지 않게 해 주세요.\n"
                "- 한 사람만 나온 사진을 사용해 주세요."
            )
    return image_file


def _split_title(title: str) -> tuple[str, str]:
    """'이마 (초년운)' -> ('이마', '초년운')"""
    m = re.match(r"^(.*?)\s*\((.+)\)\s*$", title)
    return (m.group(1), m.group(2)) if m else (title, "")


def render_fortunes(fortunes: list[tuple[str, str, str]]) -> None:
    for i in range(0, len(fortunes), 2):
        cols = st.columns(2)
        for col, (name, stars, desc) in zip(cols, fortunes[i : i + 2]):
            col.markdown(
                f'<div class="gs-fortune">'
                f'<div class="gs-f-name">{html.escape(name)}</div>'
                f'<div class="gs-f-stars">{html.escape(stars)}</div>'
                f'<div class="gs-f-desc">{html.escape(desc)}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


def render_result(result: str, image_bytes: bytes) -> None:
    if is_no_face(result):  # F-7
        _, mid, _ = st.columns([1, 2, 1])
        mid.image(image_bytes, width="stretch")
        st.warning("얼굴을 찾지 못했습니다. 얼굴이 잘 보이는 정면 사진을 올려주세요.")
        return

    parsed = parse_result(result)

    col_img, col_first = st.columns([1, 1.4], gap="medium")
    with col_img:
        st.image(image_bytes, width="stretch")
    with col_first:
        st.markdown('<div class="gs-section">첫인상</div>', unsafe_allow_html=True)
        if parsed:
            st.markdown(parsed["first"])
        else:
            st.caption("관상 풀이가 아래에 있습니다.")

    if parsed is None:  # 형식이 다르면 전체 마크다운을 그대로 보여준다 (fallback)
        with st.container(border=True):
            st.markdown(result)
        return

    st.markdown('<div class="gs-section">부위별 관상 풀이</div>', unsafe_allow_html=True)
    features = parsed["features"]
    for i in range(0, len(features), 2):
        cols = st.columns(2)
        for col, (title, body) in zip(cols, features[i : i + 2]):
            name, sub = _split_title(title)
            sub_html = f'<span class="gs-feature-sub">{html.escape(sub)}</span>' if sub else ""
            with col.container(border=True, height="stretch"):
                st.markdown(
                    f'<div class="gs-feature-title">{html.escape(name)}{sub_html}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(body)

    st.markdown('<div class="gs-section">운세 요약</div>', unsafe_allow_html=True)
    if parsed["fortunes"]:
        render_fortunes(parsed["fortunes"])
    elif parsed["fortune_raw"]:
        st.markdown(parsed["fortune_raw"])

    st.markdown('<div class="gs-section">한 줄 조언</div>', unsafe_allow_html=True)
    advice = parsed["advice"].replace("**", "").strip()
    st.markdown(
        f'<div class="gs-advice">💡 {html.escape(advice)}</div>', unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def render_body() -> None:
    if not get_api_key():
        st.error(
            "OpenAI API 키가 설정되지 않았습니다. "
            "로컬에서는 `.env` 파일에, Streamlit Cloud에서는 Secrets에 "
            "`OPENAI_API_KEY`를 등록해 주세요."
        )
        return

    image_file = get_image_input()
    if image_file is None:
        return

    if image_file.size > MAX_FILE_SIZE:  # F-1
        st.error(
            f"파일 용량이 너무 큽니다 ({image_file.size / 1024 / 1024:.1f}MB). "
            "10MB 이하의 사진을 올려주세요."
        )
        return

    image_bytes = image_file.getvalue()
    image_hash = hashlib.sha256(image_bytes).hexdigest()

    try:
        base64_image = preprocess_image(image_bytes)
    except ValueError:
        st.error("이미지 파일을 읽을 수 없습니다. JPG, PNG, WEBP 형식의 사진을 올려주세요.")
        return
    preview = base64.b64decode(base64_image)  # 회전 보정된 이미지 (F-3)

    if st.session_state.analyzing and st.session_state.pending_hash == image_hash:
        render_analyze_button(image_hash, preview, disabled=True)
        try:
            # 중단된 이전 실행이 이미 결과를 저장했다면 다시 호출하지 않는다
            if image_hash not in st.session_state.results:
                with st.spinner("관상가가 얼굴을 살펴보는 중입니다..."):
                    st.session_state.results[image_hash] = analyze_face(base64_image)
        except Exception as exc:
            st.session_state.last_error = (image_hash, friendly_error(exc))
        finally:
            st.session_state.analyzing = False
            st.session_state.pending_hash = None
        st.rerun()

    # 다른 이미지로 바뀌는 등 진행 중이던 분석이 중단되면 상태를 풀어준다
    if st.session_state.analyzing:
        st.session_state.analyzing = False
        st.session_state.pending_hash = None
        st.rerun()

    cached = st.session_state.results.get(image_hash)
    if cached is not None:
        render_result(cached, preview)
        st.write("")
        st.button("🔄 다시 보기", on_click=reset, width="stretch")
        return

    render_analyze_button(image_hash, preview, disabled=False)

    last_error = st.session_state.last_error
    if last_error and last_error[0] == image_hash:  # F-9: 오류 안내 후 같은 버튼으로 재시도
        st.error(last_error[1])


def render_analyze_button(image_hash: str, preview: bytes, disabled: bool) -> None:
    """미리보기와 분석 버튼. 분석 중에도 같은 위치·key로 비활성 버튼을 그려
    이전 실행에서 남은 버튼이 눌리는 것을 막는다 (중복 API 호출 방지)."""
    _, mid, _ = st.columns([1, 2, 1])
    mid.image(preview, caption="업로드한 사진", width="stretch")  # F-3
    has_error = bool(st.session_state.last_error) and st.session_state.last_error[0] == image_hash
    st.button(
        "🔁 다시 시도" if has_error and not disabled else "🔮 관상 보기",
        key="analyze_button",
        type="primary",
        on_click=start_analysis,
        args=(image_hash,),
        disabled=disabled,
        width="stretch",
    )


def main() -> None:
    st.set_page_config(page_title="AI 관상가", page_icon="🔮", layout="centered")
    init_state()
    inject_css()
    render_header()
    render_body()
    render_footer()


if __name__ == "__main__":
    main()
