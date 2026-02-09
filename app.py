# app.py
import os
import json
import time
import datetime as dt
from typing import Dict, Any, Optional, Tuple, List

import requests
import pandas as pd
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")

# -----------------------------
# Helpers: API calls
# -----------------------------
def get_weather(city: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    OpenWeatherMap 현재 날씨 가져오기 (한국어, 섭씨)
    실패 시 None 반환, timeout=10
    """
    if not api_key:
        return None
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key.strip(),
            "units": "metric",
            "lang": "kr",
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        weather = (data.get("weather") or [{}])[0]
        main = data.get("main") or {}

        return {
            "city": data.get("name", city),
            "temp_c": main.get("temp"),
            "feels_like_c": main.get("feels_like"),
            "humidity": main.get("humidity"),
            "desc": weather.get("description"),
            "icon": weather.get("icon"),
        }
    except Exception:
        return None


def _extract_breed_from_dog_url(url: str) -> str:
    """
    Dog CEO 이미지 URL에서 품종 추출
    예: https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg -> hound (afghan)
    """
    try:
        parts = url.split("/breeds/")[1].split("/")
        breed_part = parts[0]  # e.g., "hound-afghan" or "retriever-golden"
        tokens = breed_part.split("-")
        if len(tokens) == 1:
            return tokens[0]
        # sub-breed 포함: "hound-afghan" -> "hound (afghan)"
        return f"{tokens[0]} ({' '.join(tokens[1:])})"
    except Exception:
        return "unknown"


def get_dog_image() -> Optional[Dict[str, str]]:
    """
    Dog CEO에서 랜덤 강아지 사진 URL과 품종 가져오기
    실패 시 None 반환, timeout=10
    """
    try:
        url = "https://dog.ceo/api/breeds/image/random"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("status") != "success":
            return None
        img_url = data.get("message")
        if not img_url:
            return None
        breed = _extract_breed_from_dog_url(img_url)
        return {"url": img_url, "breed": breed}
    except Exception:
        return None


def _openai_generate_text(api_key: str, model: str, system: str, user: str) -> Optional[str]:
    """
    OpenAI 호출 (가능하면 최신 SDK -> 실패하면 HTTP fallback)
    실패 시 None
    """
    if not api_key:
        return None

    # 1) Try official SDK (new / old)
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key.strip())

        # Prefer Responses API if available
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            # responses API output parse
            text = ""
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) in ("output_text", "text"):
                        text += getattr(c, "text", "") or ""
            return text.strip() if text else None
        except Exception:
            # Fallback to Chat Completions
            cc = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.7,
            )
            return (cc.choices[0].message.content or "").strip() if cc and cc.choices else None
    except Exception:
        pass

    # 2) HTTP fallback (Chat Completions compatible endpoint)
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.7,
        }
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return None


def generate_report(
    habits: Dict[str, bool],
    mood: int,
    weather: Optional[Dict[str, Any]],
    dog: Optional[Dict[str, str]],
    coach_style: str,
    openai_api_key: str,
) -> Optional[str]:
    """
    습관+기분+날씨+강아지 품종을 모아서 OpenAI에 전달
    코치 스타일별 시스템 프롬프트 (스파르타=엄격, 멘토=따뜻, 게임마스터=RPG)
    출력 형식: 컨디션 등급(S~D), 습관 분석, 날씨 코멘트, 내일 미션, 오늘의 한마디
    모델: gpt-5-mini
    """
    style_system_prompts = {
        "스파르타 코치": (
            "너는 엄격하지만 공정한 코치다. 변명은 허용하지 않는다. "
            "짧고 단호하게, 실행 지침 위주로 말한다."
        ),
        "따뜻한 멘토": (
            "너는 다정하고 공감 능력이 뛰어난 멘토다. "
            "사용자가 스스로를 탓하지 않도록 돕고, 작은 성취를 칭찬하며 다음 행동을 부드럽게 제안한다."
        ),
        "게임 마스터": (
            "너는 RPG 게임 마스터다. 사용자의 하루를 퀘스트/스탯/보상 관점으로 해석한다. "
            "유머를 섞되 실천 가능한 미션을 준다."
        ),
    }

    system = (
        f"{style_system_prompts.get(coach_style, style_system_prompts['따뜻한 멘토'])}\n\n"
        "아래 형식을 반드시 지켜 한국어로 출력해라.\n"
        "형식:\n"
        "1) 컨디션 등급: S/A/B/C/D\n"
        "2) 습관 분석: (잘한 점 2개 + 개선 2개)\n"
        "3) 날씨 코멘트: (날씨를 활용한 조언 1~2문장)\n"
        "4) 내일 미션: (3개, 체크박스처럼 '- [ ]'로)\n"
        "5) 오늘의 한마디: (짧고 강렬하게 1문장)\n\n"
        "등급 기준 힌트: 달성률이 높고 기분이 좋으면 상향, 낮으면 하향. "
        "단, 과장하지 말고 균형 있게 판단해라."
    )

    checked = [k for k, v in habits.items() if v]
    unchecked = [k for k, v in habits.items() if not v]
    achievement = round(len(checked) / max(1, len(habits)) * 100)

    weather_line = "날씨 정보: 없음"
    if weather:
        weather_line = (
            f"날씨 정보: {weather.get('city')} / {weather.get('desc')} / "
            f"{weather.get('temp_c')}°C(체감 {weather.get('feels_like_c')}°C) / 습도 {weather.get('humidity')}%"
        )

    dog_line = "강아지: 정보 없음"
    if dog:
        dog_line = f"강아지: 품종={dog.get('breed')}"

    user = (
        "다음은 사용자의 오늘 체크인 데이터다.\n"
        f"- 달성률: {achievement}%\n"
        f"- 완료 습관: {', '.join(checked) if checked else '없음'}\n"
        f"- 미완료 습관: {', '.join(unchecked) if unchecked else '없음'}\n"
        f"- 기분(1~10): {mood}\n"
        f"- {weather_line}\n"
        f"- {dog_line}\n\n"
        "요구 형식대로 리포트를 작성해라."
    )

    return _openai_generate_text(
        api_key=openai_api_key,
        model="gpt-5-mini",
        system=system,
        user=user,
    )


# -----------------------------
# Session state init
# -----------------------------
if "records" not in st.session_state:
    st.session_state.records: List[Dict[str, Any]] = []

if "demo_loaded" not in st.session_state:
    st.session_state.demo_loaded = False

def _make_demo_records() -> List[Dict[str, Any]]:
    """
    데모용 6일 샘플 데이터
    """
    today = dt.date.today()
    demo = []
    # 지난 6일
    for i in range(6, 0, -1):
        d = today - dt.timedelta(days=i)
        # 약간의 변동이 있는 샘플
        checks = (i % 6)  # 0~5 변동
        checks = min(5, max(0, checks))
        mood = 4 + (i % 7)  # 4~10
        demo.append(
            {
                "date": d.isoformat(),
                "checked": int(checks),
                "achievement": round(checks / 5 * 100),
                "mood": int(mood),
            }
        )
    return demo

def _upsert_today_record(checked: int, achievement: int, mood: int) -> None:
    today_str = dt.date.today().isoformat()
    found = False
    for r in st.session_state.records:
        if r.get("date") == today_str:
            r.update({"checked": checked, "achievement": achievement, "mood": mood})
            found = True
            break
    if not found:
        st.session_state.records.append(
            {"date": today_str, "checked": checked, "achievement": achievement, "mood": mood}
        )
    # 날짜 정렬 & 최근 7개 유지
    st.session_state.records = sorted(st.session_state.records, key=lambda x: x.get("date", ""))
    if len(st.session_state.records) > 7:
        st.session_state.records = st.session_state.records[-7:]


# -----------------------------
# Sidebar: API keys
# -----------------------------
with st.sidebar:
    st.header("🔑 API 키 설정")
    openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    owm_api_key = st.text_input("OpenWeatherMap API Key", type="password", value=os.getenv("OPENWEATHERMAP_API_KEY", ""))
    st.caption("키는 로컬에서만 사용되며, 이 앱은 저장소에 키를 저장하지 않도록 설계하세요.")

# -----------------------------
# Main UI
# -----------------------------
st.title("📊 AI 습관 트래커")
st.write("오늘의 습관을 체크하고, 날씨 + 강아지 + AI 코치 리포트로 컨디션을 점검해보세요.")

# Load demo once
if not st.session_state.demo_loaded:
    st.session_state.records = _make_demo_records()
    st.session_state.demo_loaded = True

st.subheader("✅ 습관 체크인")

# Habits in 2 columns, 5 checkboxes
habit_defs = [
    ("🌅 기상 미션", "wake"),
    ("💧 물 마시기", "water"),
    ("📚 공부/독서", "study"),
    ("🏃 운동하기", "workout"),
    ("😴 수면", "sleep"),
]

left, right = st.columns(2, gap="large")

habits: Dict[str, bool] = {}
# Place 3 left, 2 right for balance
for idx, (label, key) in enumerate(habit_defs):
    col = left if idx in (0, 1, 2) else right
    with col:
        habits[label] = st.checkbox(label, key=f"habit_{key}")

mood = st.slider("🙂 오늘 기분은 어떤가요? (1~10)", min_value=1, max_value=10, value=6, step=1)

cities = [
    "Seoul", "Busan", "Incheon", "Daegu", "Daejeon",
    "Gwangju", "Suwon", "Ulsan", "Jeju City", "Changwon"
]
city = st.selectbox("🏙️ 도시 선택", options=cities, index=0)

coach_style = st.radio(
    "🧑‍🏫 코치 스타일",
    options=["스파르타 코치", "따뜻한 멘토", "게임 마스터"],
    horizontal=True,
)

# -----------------------------
# Achievement + Metrics + Chart
# -----------------------------
checked_count = sum(1 for v in habits.values() if v)
achievement = round(checked_count / 5 * 100)

st.subheader("📈 달성률 & 주간 흐름")

m1, m2, m3 = st.columns(3)
m1.metric("달성률", f"{achievement}%", help="오늘 체크한 습관 비율입니다.")
m2.metric("달성 습관", f"{checked_count}/5", help="체크된 습관 개수입니다.")
m3.metric("기분", f"{mood}/10", help="자기 보고 기분 점수입니다.")

# Upsert today's record to session_state for chart
_upsert_today_record(checked=checked_count, achievement=achievement, mood=mood)

df = pd.DataFrame(st.session_state.records)
if not df.empty:
    # Ensure 7 rows (demo 6 + today)
    df = df.sort_values("date")
    # Bar chart for achievement
    chart_df = df.set_index("date")[["achievement"]]
    st.bar_chart(chart_df, height=240)

# -----------------------------
# Result area: Weather + Dog + AI report
# -----------------------------
st.subheader("🧠 AI 코치 리포트")

gen = st.button("컨디션 리포트 생성", type="primary")

weather_data = None
dog_data = None
report_text = None

if gen:
    with st.spinner("날씨, 강아지, AI 리포트를 가져오는 중..."):
        # Fetch weather & dog first (fast)
        weather_data = get_weather(city, owm_api_key)
        dog_data = get_dog_image()

        # Generate report
        report_text = generate_report(
            habits=habits,
            mood=mood,
            weather=weather_data,
            dog=dog_data,
            coach_style=coach_style,
            openai_api_key=openai_api_key,
        )

    # Display cards in 2 columns
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("### 🌤️ 오늘의 날씨")
        if weather_data:
            icon = weather_data.get("icon")
            icon_url = f"https://openweathermap.org/img/wn/{icon}@2x.png" if icon else None
            if icon_url:
                st.image(icon_url, width=80)
            st.write(f"**도시**: {weather_data.get('city')}")
            st.write(f"**상태**: {weather_data.get('desc')}")
            st.write(f"**기온**: {weather_data.get('temp_c')}°C (체감 {weather_data.get('feels_like_c')}°C)")
            st.write(f"**습도**: {weather_data.get('humidity')}%")
        else:
            st.info("날씨를 가져오지 못했어요. OpenWeatherMap API Key와 도시 이름을 확인해 주세요.")

    with c2:
        st.markdown("### 🐶 오늘의 강아지")
        if dog_data:
            st.image(dog_data["url"], use_container_width=True)
            st.caption(f"품종: {dog_data.get('breed', 'unknown')}")
        else:
            st.info("강아지 이미지를 가져오지 못했어요. 잠시 후 다시 시도해 주세요.")

    st.markdown("### 🧾 AI 컨디션 리포트")
    if report_text:
        st.write(report_text)
    else:
        st.warning("리포트를 생성하지 못했어요. OpenAI API Key 또는 네트워크 상태를 확인해 주세요.")

    # Share text
    st.markdown("### 📣 공유용 텍스트")
    done_list = [k for k, v in habits.items() if v]
    weather_short = (
        f"{weather_data.get('desc')} {weather_data.get('temp_c')}°C" if weather_data else "날씨 없음"
    )
    dog_short = dog_data.get("breed") if dog_data else "강아지 없음"

    share_text = (
        f"📊 AI 습관 트래커 체크인 ({dt.date.today().isoformat()})\n"
        f"- 달성률: {achievement}% ({checked_count}/5)\n"
        f"- 완료: {', '.join(done_list) if done_list else '없음'}\n"
        f"- 기분: {mood}/10\n"
        f"- 날씨: {city} / {weather_short}\n"
        f"- 강아지: {dog_short}\n\n"
        f"[AI 리포트]\n{report_text or '(리포트 생성 실패)'}\n"
    )
    st.code(share_text, language="text")

# -----------------------------
# Footer: API 안내
# -----------------------------
with st.expander("📌 API 안내 / 문제 해결"):
    st.markdown(
        """
- **OpenAI API Key**: 리포트 생성에 사용됩니다. (모델: `gpt-5-mini`)
- **OpenWeatherMap API Key**: 선택한 도시의 현재 날씨를 가져옵니다. (`units=metric`, `lang=kr`)
- **Dog CEO API**: 랜덤 강아지 이미지를 가져옵니다. (키 불필요)

**팁**
- 날씨가 안 나오면: OpenWeatherMap 키가 활성화되어 있는지, 도시 표기가 정확한지 확인해 주세요.
- 리포트가 안 나오면: OpenAI 키가 맞는지, 사용량/권한(결제/쿼터)을 확인해 주세요.
- 네트워크 환경에 따라 간헐적으로 실패할 수 있으며, 이 앱은 요청에 `timeout=10`을 적용합니다.
        """
    )
